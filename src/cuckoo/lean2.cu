// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

// The edge-trimming memory optimization is due to Dave Andersen
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html

#include <stdint.h>
#include <string.h>
#include "cuckoo.h"
#include "../crypto/siphash.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <set>

// algorithm parameters
#ifndef PART_BITS
// #bits used to partition edge set processing to save memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two, making twice_set the
// same size as shrinkingset at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) == sizeof(twice_set), so
// CUCKOO_SIZE * sizeof(u64) == TWICE_WORDS * sizeof(u32)
// CUCKOO_SIZE * 2 == TWICE_WORDS
// (NNODES >> IDXSHIFT) * 2 == 2 * ONCE_BITS / 32
// NNODES >> IDXSHIFT == NEDGES >> PART_BITS >> 5
// IDXSHIFT == 1 + PART_BITS + 5
#define IDXSHIFT (PART_BITS + 6)
#endif

#define EDGEBITS 30
#define NODEBITS (EDGEBITS + 1)
#define NNODES (2 * NEDGES)
#define NODEMASK (NNODES-1)

// grow with cube root of size, hardly affected by trimming
#define MAXPATHLEN (8 << (NODEBITS/3))

#ifndef EDGE_BLOCK_BITS
#define EDGE_BLOCK_BITS 6
#endif
#define EDGE_BLOCK_SIZE (1 << EDGE_BLOCK_BITS)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf (stderr, "GPUassert: %s %s %d\n", cudaGetErrorString (code), file, line);
		if (abort)
			exit (code);
	}
}

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset
{
  public:
	u32 * bits;
	__device__ void reset (word_t n)
	{
		bits[n / 32] |= 1 << (n % 32);
	}
	__device__ bool test (word_t n) const
	{
		return !((bits[n / 32] >> (n % 32)) & 1);
	}
	__device__ u64 block (word_t n) const
	{
		u64 r = *(u64 *) & bits[n / 32];
		//  return ~bits[n/32;
		  return ~r;
	}
};

#define PART_MASK ((1 << PART_BITS) - 1)
#define ONCE_BITS (NEDGES >> PART_BITS)
#define TWICE_WORDS ((2 * ONCE_BITS) / 32)

class twice_set
{
  public:
	u32 * bits;
	__device__ void reset ()
	{
		memset (bits, 0, TWICE_WORDS * sizeof (u32));
	}
	__device__ void set (word_t u)
	{
		word_t idx = u / 16;
		u32 bit = 1 << (2 * (u % 16));
		u32 old = atomicOr (&bits[idx], bit);
		u32 bit2 = bit << 1;
		if ((old & (bit2 | bit)) == bit)
			atomicOr (&bits[idx], bit2);
	}
	__device__ u32 test (word_t u) const
	{
		return (bits[u / 16] >> (2 * (u % 16))) & 2;
	}
};

#define CUCKOO_SIZE (NNODES >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by NODEBITS
#define KEYBITS (64-NODEBITS)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

class cuckoo_hash
{
  public:
	u64 * cuckoo;

	cuckoo_hash ()
	{
		cuckoo = (u64 *) calloc (CUCKOO_SIZE, sizeof (u64));
		assert (cuckoo != 0);
	}
	 ~cuckoo_hash ()
	{
		free (cuckoo);
	}
	void set (word_t u, word_t v)
	{
		u64 niew = (u64) u << NODEBITS | v;
		for (word_t ui = u >> IDXSHIFT;; ui = (ui + 1) & CUCKOO_MASK)
		{
#ifdef ATOMIC
			u64 old = 0;
			if (cuckoo[ui].compare_exchange_strong (old, niew, std::memory_order_relaxed))
				return;
			if ((old >> NODEBITS) == (u & KEYMASK))
			{
				cuckoo[ui].store (niew, std::memory_order_relaxed);
#else
			u64 old = cuckoo[ui];
			if (old == 0 || (old >> NODEBITS) == (u & KEYMASK))
			{
				cuckoo[ui] = niew;
#endif
				return;
			}
		}
	}
	word_t operator[] (word_t u) const
	{
		for (word_t ui = u >> IDXSHIFT;; ui = (ui + 1) & CUCKOO_MASK)
		{
#ifdef ATOMIC
			u64 cu = cuckoo[ui].load (std::memory_order_relaxed);
#else
			u64 cu = cuckoo[ui];
#endif
			if (!cu)
				  return 0;
			if ((cu >> NODEBITS) == (u & KEYMASK))
			{
				assert (((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
				return (word_t) (cu & NODEMASK);
			}
		}
	}
};

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

class cuckoo_ctx
{
  public:
	siphash_keys sip_keys;
	shrinkingset alive;
	twice_set nonleaf;
	int nthreads;

	  cuckoo_ctx (const u32 n_threads)
	{
		nthreads = n_threads;
	}
	void setheadernonce (char *headernonce, const u32 nonce)
	{
		((u32 *) headernonce)[HEADERLEN / sizeof (u32) - 1] = htole32 (nonce);	// place nonce at end
		setheader (headernonce, HEADERLEN, &sip_keys);
	}
};

__device__ u64 dipblock (const siphash_keys & keys, const word_t edge, u64 * buf)
{
	u64 v0 = keys.k0, v1 = keys.k1, v2 = keys.k2, v3 = keys.k3;
	word_t edge0 = edge & ~EDGE_BLOCK_MASK;
	u32 i;
	for (i = 0; i < EDGE_BLOCK_MASK; i++)
	{
		word_t nonce = edge0 + i;
		v3 ^= nonce;
		SIPROUND;
		SIPROUND;
		v0 ^= nonce;
		v2 ^= 0xff;
		SIPROUND;
		SIPROUND;
		SIPROUND;
		SIPROUND;
		buf[i] = (v0 ^ v1) ^ (v2 ^ v3);
	}
	word_t nonce = edge0 + i;
	v3 ^= nonce;
	SIPROUND;
	SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND;
	SIPROUND;
	SIPROUND;
	SIPROUND;

	buf[i] = 0;
	return (v0 ^ v1) ^ (v2 ^ v3);
}

/*
__global__ void count_node_deg(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  twice_set &nonleaf = ctx->nonleaf;
  siphash_keys sip_keys = ctx->sip_keys; // local copy sip context; 2.5% speed gain
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (u32 block = id*64; block < NEDGES; block += ctx->nthreads*64) {
    u64 alive32 = alive.block(block);
    for (u32 nonce = block-1; alive32; ) { // -1 compensates for 1-based ffs
      u32 ffs = __ffsll(alive32);
      nonce += ffs; alive32 >>= ffs;
      u32 u = dipnode(sip_keys, nonce, uorv);
      if ((u & PART_MASK) == part) {
        nonleaf.set(u >> PART_BITS);
      }
    }
  }
}*/

__global__ void count_node_deg (cuckoo_ctx * ctx, u32 uorv, u32 part, unsigned long long int *hash_count, int* nonce_hash_count, unsigned long long int *rw_count)
{
	shrinkingset & alive = ctx->alive;
	twice_set & nonleaf = ctx->nonleaf;
	siphash_keys sip_keys = ctx->sip_keys;	// local copy sip context; 2.5% speed gain
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	u64 buf[64];
	for (u32 block = id * 64; block < NEDGES; block += ctx->nthreads * 64)
	{
		u64 alive64 = alive.block (block);
		u64 last = 0;
		if(alive64) // if commenting this code, will more quickly
		{
			last = dipblock (sip_keys, block, buf);
	//		atomicAdd(hash_count, 64);
	//		atomicAdd(rw_count, 64);
		//	atomicAdd(nonce_hash_count + block/64, 1);
		}
		for (u32 nonce = block - 1; alive64;)
		{						// -1 compensates for 1-based ffs
			u32 ffs = __ffsll (alive64);
			nonce += ffs;
			alive64 >>= ffs;

			u64 edge = buf[nonce - block] ^ last;
			u32 u = (edge >> (uorv ? 32 : 0)) & EDGEMASK;

			if ((u & PART_MASK) == part)
			{
				nonleaf.set (u >> PART_BITS);
			//	atomicAdd(rw_count, 1);
			}

		}
	}
}

__global__ void kill_leaf_edges (cuckoo_ctx * ctx, u32 uorv, u32 part, unsigned long long int* hash_count, int* nonce_hash_count, unsigned long long int *rw_count)
{
	shrinkingset & alive = ctx->alive;
	twice_set & nonleaf = ctx->nonleaf;
	siphash_keys sip_keys = ctx->sip_keys;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	u64 buf[64];
	for (u32 block = id * 64; block < NEDGES; block += ctx->nthreads * 64)
	{
		u64 alive64 = alive.block (block);
		u64 last = 0;
		if(alive64)
		{
			last = dipblock (sip_keys, block, buf);
		//	atomicAdd(hash_count, 64);
		//	atomicAdd(rw_count, 64);
		//	atomicAdd(nonce_hash_count + block/64, 1);
		}
		for (u32 nonce = block - 1; alive64;)
		{						// -1 compensates for 1-based ffs
			u32 ffs = __ffsll (alive64);
			nonce += ffs;
			alive64 >>= ffs;

			u64 edge = buf[nonce - block] ^ last;
			u32 u = (edge >> (uorv ? 32 : 0)) & EDGEMASK;

			if ((u & PART_MASK) == part)
			{
				if (!nonleaf.test (u >> PART_BITS))
				{
					alive.reset (nonce);
				//	atomicAdd(rw_count, 1);
				}
			}
		}
	}
}

/*
__global__ void kill_leaf_edges(cuckoo_ctx *ctx, u32 uorv, u32 part) {
  shrinkingset &alive = ctx->alive;
  twice_set &nonleaf = ctx->nonleaf;
  siphash_keys sip_keys = ctx->sip_keys;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (u32 block = id*64; block < NEDGES; block += ctx->nthreads*64) {
    u64 alive32 = alive.block(block);
    for (u32 nonce = block-1; alive32; ) { // -1 compensates for 1-based ffs
      u32 ffs = __ffsll(alive32);
      nonce += ffs; alive32 >>= ffs;
      u32 u = dipnode(sip_keys, nonce, uorv);
      if ((u & PART_MASK) == part) {
        if (!nonleaf.test(u >> PART_BITS)) {
          alive.reset(nonce);
        }
      }
    }
  }
}
*/
u32 path (cuckoo_hash & cuckoo, word_t u, word_t * us)
{
	u32 nu;
	for (nu = 0; u; u = cuckoo[u])
	{
		if (nu >= MAXPATHLEN)
		{
			while (nu-- && us[nu] != u) ;
			if (nu == ~0)
				printf ("maximum path length exceeded\n");
			else
				printf ("illegal % 4d-cycle\n", MAXPATHLEN - nu);
			exit (0);
		}
		us[nu++] = u;
	}
	return nu - 1;
}

u64 sipblock (siphash_keys & keys, const word_t edge, u64 * buf)
{
	u64 v0 = keys.k0, v1 = keys.k1, v2 = keys.k2, v3 = keys.k3;
	word_t edge0 = edge & ~EDGE_BLOCK_MASK;
	u32 i;
	for (i = 0; i < EDGE_BLOCK_MASK; i++)
	{
		word_t nonce = edge0 + i;
		v3 ^= nonce;
		SIPROUND;
		SIPROUND;
		v0 ^= nonce;
		v2 ^= 0xff;
		SIPROUND;
		SIPROUND;
		SIPROUND;
		SIPROUND;
		buf[i] = (v0 ^ v1) ^ (v2 ^ v3);
	}
	word_t nonce = edge0 + i;
	v3 ^= nonce;
	SIPROUND;
	SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND;
	SIPROUND;
	SIPROUND;
	SIPROUND;

	buf[i] = 0;
	return (v0 ^ v1) ^ (v2 ^ v3);
}

typedef std::pair < word_t, word_t > edge;

#include <unistd.h>

int main (int argc, char **argv)
{
	int nthreads = 16384;
	int trims = 32;
	int tpb = 0;
	int nonce = 0;
	int range = 1;
	const char *header = "";
	int c;
	while ((c = getopt (argc, argv, "h:n:m:r:t:p:")) != -1)
	{
		switch (c)
		{
		case 'h':
			header = optarg;
			break;
		case 'n':
			nonce = atoi (optarg);
			break;
		case 'm':
			trims = atoi (optarg);
			break;
		case 't':
			nthreads = atoi (optarg);
			break;
		case 'p':
			tpb = atoi (optarg);
			break;
		case 'r':
			range = atoi (optarg);
			break;
		}
	}
	if (!tpb)					// if not set, then default threads per block to roughly square root of threads
		for (tpb = 1; tpb * tpb < nthreads; tpb *= 2) ;

	printf ("Looking for %d-cycle on cuckoo%d(\"%s\",%d", PROOFSIZE, NODEBITS, header, nonce);
	if (range > 1)
		printf ("-%d", nonce + range - 1);
	printf (") with 50%% edges, %d trims, %d threads %d per block\n", trims, nthreads, tpb);

	cuckoo_ctx ctx (nthreads);

	char headernonce[HEADERLEN];
	u32 hdrlen = strlen (header);
	memcpy (headernonce, header, hdrlen);
	memset (headernonce + hdrlen, 0, sizeof (headernonce) - hdrlen);

	u64 edgeBytes = NEDGES / 8, nodeBytes = TWICE_WORDS * sizeof (u32);
	checkCudaErrors (cudaMalloc ((void **) &ctx.alive.bits, edgeBytes));
	checkCudaErrors (cudaMalloc ((void **) &ctx.nonleaf.bits, nodeBytes));

	int edgeUnit = 0, nodeUnit = 0;
	u64 eb = edgeBytes, nb = nodeBytes;
	for (; eb >= 1024; eb >>= 10)
		edgeUnit++;
	for (; nb >= 1024; nb >>= 10)
		nodeUnit++;
	printf ("Using %d%cB edge and %d%cB node memory.\n", (int) eb, " KMGT"[edgeUnit], (int) nb, " KMGT"[nodeUnit]);

	cuckoo_ctx *device_ctx;
	checkCudaErrors (cudaMalloc ((void **) &device_ctx, sizeof (cuckoo_ctx)));

	cudaEvent_t start, stop;
	checkCudaErrors (cudaEventCreate (&start));
	checkCudaErrors (cudaEventCreate (&stop));
	unsigned long long k0 = 0xa34c6a2bdaa03a14ULL;
	unsigned long long k1 = 0xd736650ae53eee9eULL;
	unsigned long long k2 = 0x9a22f05e3bffed5eULL;
	unsigned long long k3 = 0xb8d55478fa3a606dULL;
	ctx.sip_keys.k0 = k0;
	ctx.sip_keys.k1 = k1;
	ctx.sip_keys.k2 = k2;
	ctx.sip_keys.k3 = k3;
	printf("%lu, %lu, %lu, %lu\n", ctx.sip_keys.k0, ctx.sip_keys.k1, ctx.sip_keys.k2, ctx.sip_keys.k3);

	unsigned long hash_count = 0;
	unsigned long long int*dev_hash_count;
	cudaMalloc((void**)&dev_hash_count, sizeof(unsigned long long int));
	int *nonce_hash_count = (int*)malloc(sizeof(int) * NEDGES/64);
	int *dev_nonce_hash_count;
	unsigned long long int  *dev_rw_count;
	cudaMalloc((void**)&dev_nonce_hash_count, sizeof(int) * NEDGES/64);
	cudaMalloc((void**)&dev_rw_count, sizeof(unsigned long long int));
	cudaMemcpy(dev_rw_count, &hash_count, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

	FILE *fcount = fopen("cuckaroo_lean.txt", "w");
	range = 1;
	for (int r = 0; r < range; r++)
	{
		clock_t cstart = clock ();
		cudaEventRecord (start, NULL);
		checkCudaErrors (cudaMemset (ctx.alive.bits, 0, edgeBytes));
		ctx.setheadernonce (headernonce, nonce + r);
		cudaMemcpy (device_ctx, &ctx, sizeof (cuckoo_ctx), cudaMemcpyHostToDevice);
	//	trims += 1;
		for (u32 round = 0; round < trims; round++)
		{
			for (u32 uorv = 0; uorv < 2; uorv++)
			{
				for (u32 part = 0; part <= PART_MASK; part++)
				{
					checkCudaErrors (cudaMemset (ctx.nonleaf.bits, 0, nodeBytes));
					count_node_deg <<< nthreads / tpb, tpb >>> (device_ctx, uorv, part, dev_hash_count, dev_nonce_hash_count, dev_rw_count);
					kill_leaf_edges <<< nthreads / tpb, tpb >>> (device_ctx, uorv, part, dev_hash_count, dev_nonce_hash_count, dev_rw_count);
				}
			}
		}

		u64 *bits;
		bits = (u64 *) calloc (NEDGES / 64, sizeof (u64));
		assert (bits != 0);
		cudaMemcpy (bits, ctx.alive.bits, (NEDGES / 64) * sizeof (u64), cudaMemcpyDeviceToHost);

		cudaEventRecord (stop, NULL);
		cudaEventSynchronize (stop);
		float duration;
		cudaEventElapsedTime (&duration, start, stop);
		u32 cnt = 0;
		for (int i = 0; i < NEDGES / 64; i++)
			cnt += __builtin_popcountll (~bits[i]);
		printf ("trim result : %u\n", cnt);
		u32 load = (u32) (100L * cnt / CUCKOO_SIZE);
		cudaMemcpy(&hash_count, dev_hash_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
//		cudaMemcpy(nonce_hash_count, dev_nonce_hash_count, sizeof(int) * NEDGES/64, cudaMemcpyDeviceToHost);
		unsigned long rw_count = 0;
		cudaMemcpy(&rw_count, dev_rw_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
		printf ("nonce %d: %d trims completed in %.3f seconds final load %d%%, hash count = %lu, rw_count = %lu\n", nonce + r, trims, duration / 1000.0f, load, hash_count, rw_count);

		/*FILE*fp = fopen("nonce_hash_count2.txt", "w+");
		if(fp == NULL) return;
		for(int i = 0; i < NEDGES/64; i++){
			//for(int j = 0; j < 64; j++)
			{
			//	fprintf(fp, "%d %d\n", i*64, nonce_hash_count[i]);		
			}
		}
		fclose(fp);
		cudaFree(dev_nonce_hash_count);
		cudaFree(dev_hash_count);
*/
		if (load >= 90)
		{
			printf ("overloaded! exiting...");
			exit (0);
		}

		cuckoo_hash & cuckoo = *(new cuckoo_hash ());
		word_t us[MAXPATHLEN], vs[MAXPATHLEN];

		edge *trimeedges = (edge *) malloc (sizeof (edge) * cnt);
		u64 buf[64];
		for (word_t block = 0, i = 0; block < NEDGES; block += 64)
		{
			u64 alive64 = ~bits[block / 64];
			if (!alive64)
				continue;
			const u64 last = sipblock (ctx.sip_keys, block, buf);
			for (word_t nonce = block - 1; alive64;)
			{
				u32 ffs = __builtin_ffsll (alive64);
				nonce += ffs;
				if (ffs == 64) alive64 = 0;
				else
				alive64 >>= ffs;
				if (nonce - block < 64)
				{
					u64 one_edge = buf[nonce - block] ^ last;
					word_t u0 = one_edge & EDGEMASK;
					word_t v0 = (one_edge >> 32) & EDGEMASK;
					edge newedge (u0, v0);
					trimeedges[i++] = newedge;
				}
			}
		}
		printf("find cycle :\n");
		for (int i = 0; i < cnt; i++)
		{
			word_t u0 = trimeedges[i].first;	//one_edge & EDGEMASK;
			word_t v0 = trimeedges[i].second;	//(one_edge >> 32) & EDGEMASK;
			if (u0)
			{
				u32 nu = path (cuckoo, u0, us), nv = path (cuckoo, v0, vs);
				if (us[nu] == vs[nv])
				{
					u32 min = nu < nv ? nu : nv;
					for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
					u32 len = nu + nv + 1;
					printf ("%4d-cycle found at %d\n", len, 0);
					if (len == PROOFSIZE)
					{
						printf ("Solution");
						std::set < edge > cycle;
						u32 n = 0;
						cycle.insert (edge (*us, *vs));
						while (nu--)
							cycle.insert (edge (us[(nu + 1) & ~1], us[nu | 1]));	// u's in even position; v's in odd
						while (nv--)
							cycle.insert (edge (vs[nv | 1], vs[(nv + 1) & ~1]));	// u's in odd position; v's in even
						u64 tmpbuf[64];
						for (int j = 0; j < cnt; j++)
						{
							edge e = trimeedges[j];
							if (cycle.find (e) != cycle.end ())
							{
								//printf(" %jx", (uintmax_t)nce);
								if (PROOFSIZE > 2)
									cycle.erase (e);
								n++;
							}
						}
						assert (n == PROOFSIZE);
						printf ("\n");
					}
				}
				else if (nu < nv)
				{
					while (nu--)
						cuckoo.set (us[nu + 1], us[nu]);
					cuckoo.set (u0, v0);
				}
				else
				{
					while (nv--)
						cuckoo.set (vs[nv + 1], vs[nv]);
					cuckoo.set (v0, u0);
				}
			}
		}
		
		clock_t cend = clock ();
		printf ("all time : %.4f\n", (double) (cend - cstart) / CLOCKS_PER_SEC);
		fprintf(fcount, "%d %d %.3f %.3f\n", trims, cnt, duration/1000.0f, (double)(cend-cstart)/CLOCKS_PER_SEC - duration/1000.0f);
		free(trimeedges);
		free(bits);
	}
	fclose(fcount);
	checkCudaErrors (cudaFree (ctx.alive.bits));
	checkCudaErrors (cudaFree (ctx.nonleaf.bits));
	return 0;
}
