#ifndef WLT_TRIMMER_H
#define WLT_TRIMMER_H
#include <string>
static const std::string kernel_source = R"(
#ifndef EDGEBITS
#define EDGEBITS 30
#endif
#ifndef PROOFSIZE
#define PROOFSIZE 42
#endif

#if EDGEBITS > 32
typedef ulong edge_t;
#else
typedef uint edge_t;
#endif
#if EDGEBITS > 31
typedef ulong node_t;
#else
typedef uint node_t;
#endif

#define NEDGES ((node_t)1 << EDGEBITS)
#define EDGEMASK ((edge_t)NEDGES - 1)

#ifndef XBITS
#define XBITS 6 //((EDGEBITS-16)/2)
#endif
#ifndef YBITS
#define YBITS 7 //((EDGEBITS-16)/2)
#endif

#define NX        (1 << (XBITS))
#define NY        (1 << (YBITS))
#define NX2       ((NX) * (NY))
#define NXY       ((NX) * (NY))
#define XMASK     ((NX) - 1)
#define YMASK     ((NY) - 1)
#define X2MASK    ((NX2) - 1)
#define YZBITS  ((EDGEBITS) - (XBITS))
#define ZBITS     ((YZBITS) - (YBITS))
#define NZ        (1 << (ZBITS))
#define COUNTERWORDS  ((NZ) / 16)
#define ZMASK     (NZ - 1)

#ifndef FLUSHA			// should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

#ifndef FLUSHB
#define FLUSHB 8
#endif

#ifndef EDGE_BLOCK_BITS
#define EDGE_BLOCK_BITS 6
#endif
#define EDGE_BLOCK_SIZE (1 << EDGE_BLOCK_BITS)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)


typedef struct {
    ulong k0;
    ulong k1;
    ulong k2;
    ulong k3;
} siphash_keys;

#define U8TO64_LE(p) ((p))
#define ROTL(x,b) (ulong)( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
    do { \
      v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
      v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
      v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
      v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
      v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
    } while(0)

#define FLUSHB2 (2 * (FLUSHB))
#define FLUSHA2  (2 * (FLUSHA))

inline ulong4
make_ulong4(ulong r1, ulong r2, ulong r3, ulong r4)
{
    return (ulong4) (r1, r2, r3, r4);
}

inline uint2
make_uint2(uint a, uint b)
{
    return (uint2) (a, b);
}

inline ulong4
Pack4edges(const uint2 e1, const uint2 e2, const uint2 e3, const uint2 e4)
{
    ulong r1 = (((ulong) e1.y << 32) | ((ulong) e1.x));
    ulong r2 = (((ulong) e2.y << 32) | ((ulong) e2.x));
    ulong r3 = (((ulong) e3.y << 32) | ((ulong) e3.x));
    ulong r4 = (((ulong) e4.y << 32) | ((ulong) e4.x));
    return make_ulong4(r1, r2, r3, r4);
}

inline ulong4
uint2_to_ulong4(uint2 v0, uint2 v1, uint2 v2, uint2 v3)
{
    return Pack4edges(v0, v1, v2, v3);
}

inline node_t
dipnode(__constant const siphash_keys * keys, edge_t nce, uint uorv)
{
    ulong nonce = 2 * nce + uorv;
    ulong v0 = (*keys).k0, v1 = (*keys).k1, v2 = (*keys).k2, v3 = (*keys).k3 ^ nonce;
    SIPROUND;
    SIPROUND;
    v0 ^= nonce;
    v2 ^= 0xff;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    return (v0 ^ v1 ^ v2 ^ v3) & EDGEMASK;
}

inline uint
endpoint(__constant const siphash_keys * sipkeys, uint nonce, int uorv)
{
    return dipnode(sipkeys, nonce, uorv);
}

inline uint
endpoint2(__constant const siphash_keys * sipkeys, uint2 nodes, int uorv)
{
    return uorv ? nodes.y : nodes.x;
}

inline uint2
make_Edge_by_node(const uint nonce, const uint2 dummy, const uint node0, const uint node1)
{
    return make_uint2(node0, node1);
}

inline uint2
make_Edge_by_edge(const uint2 edge, const uint2 dummy, const uint node0, const uint node1)
{
    return edge;
}

inline uint
make_Edge_by_nonce(const uint nonce, const uint dummy, const uint node0, const uint node1)
{
    return nonce;
}

inline void
Increase2bCounter(__local uint * ecounters, const int bucket)
{
    int word = bucket >> 5;
    unsigned char bit = bucket & 0x1F;
    uint mask = 1 << bit;

    uint old = atomic_or(ecounters + word, mask) & mask;
    if (old)
	atomic_or(ecounters + word + NZ / 32, mask);
}

inline bool
Read2bCounter(__local uint * ecounters, const int bucket)
{
    int word = bucket >> 5;
    unsigned char bit = bucket & 0x1F;
    //uint mask = 1 << bit;

    //return (ecounters[word + NZ / 32] & mask) != 0;
		return (ecounters[word + NZ/32] >> bit ) & 1;
}

inline void bitmapset(__local uint *ebitmap, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  uint mask = 1 << bit;
  uint old = atomic_or(ebitmap + word, mask) & mask;
  if(old) atomic_or(ebitmap + word + NZ/32, mask);
}

inline  bool bitmaptest(__local uint *ebitmap, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  return (ebitmap[word + NZ/32] >> bit) & 1;
}

inline ulong4
Pack8(const uint e0, const uint e1, const uint e2, const uint e3, const uint e4, const uint e5, const uint e6, const uint e7)
{
    return make_ulong4((long) e0 << 32 | e1, (long) e2 << 32 | e3, (long) e4 << 32 | e5, (long) e6 << 32 | e7);
}

inline bool
null(uint nonce)
{
    return (nonce == 0);
}

inline bool
null2(uint2 nodes)
{
    return (nodes.x == 0 && nodes.y == 0);
}


inline ulong dipblock(__constant const siphash_keys *key, const edge_t edge, ulong *buf) {
  //diphash_state shs(keys);
  siphash_keys keys = *key;
  ulong v0 = keys.k0, v1 = keys.k1, v2 = keys.k2, v3 = keys.k3;

  edge_t edge0 = edge & ~EDGE_BLOCK_MASK;
  uint i;
  for (i=0; i < EDGE_BLOCK_MASK; i++) {
    //shs.hash24(edge0 + i);
	  edge_t nonce = edge0 + i;
		v3^=nonce;
		SIPROUND; SIPROUND; SIPROUND; SIPROUND;
		v0 ^= nonce;
		v2 ^= 0xff;	
		SIPROUND; SIPROUND; SIPROUND; SIPROUND;
		SIPROUND; SIPROUND; SIPROUND; SIPROUND;

//    buf[i] = shs.xor_lanes();
		buf[i] = (v0 ^ v1) ^ (v2  ^ v3);
  }
//  shs.hash24(edge0 + i);
	edge_t nonce = edge0 + i;
	v3^=nonce;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;

//    buf[i] = shs.xor_lanes();
	buf[i] = 0;
  //return shs.xor_lanes();
	return (v0 ^ v1) ^ (v2  ^ v3);
}

__kernel void Cuckaroo_Recovery(__constant const siphash_keys *sipkeys,__global int *indexes, __constant uint2* recoveredges) {
  const int gid = get_global_id(0);//blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = get_local_id(0);//threadIdx.x;
  const int nthreads = get_global_size(0);//blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __local uint nonces[PROOFSIZE];
  ulong buf[EDGE_BLOCK_SIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    uint nonce0 = gid * loops + blk;
    const ulong last = dipblock(sipkeys, nonce0, buf);
    for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
      ulong edge = buf[i] ^ last;
      uint u = edge & EDGEMASK;
      uint v = (edge >> 32) & EDGEMASK;
      for (int p = 0; p < PROOFSIZE; p++) {
        if (recoveredges[p].y == u && recoveredges[p].x == v)
          nonces[p] = nonce0 + i;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}

#define HALF_EDGES (NEDGES >> 1)

__kernel void 
Cuckaroo_SeedA1(const int round, const int seg,
		__constant const siphash_keys* sipkeys,
		__global uint2 * __restrict__ buffer1,
		__global uint2 * __restrict__ buffer2,
		__global uint * __restrict__ indexes, 
		const int maxOut, __global ulong *one_bit_set) {
  const int group = get_group_id(0);//blockIdx.x;
  const int dim = get_local_size(0);//blockDim.x;
  const int lid = get_local_id(0);//threadIdx.x;
  const int gid = group * dim + lid;
  const int nthreads = get_global_size(0);//gridDim.x * dim;

  __local uint2 tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned
  const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
  __local int counters[NX];
  ulong buf[EDGE_BLOCK_SIZE];

  for (int row = lid; row < NX; row += dim)
    counters[row] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  const int col = group % NY;
  const int loops = HALF_EDGES / nthreads; // assuming THREADS_HAVE_EDGES checked
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
   uint nonce0 = gid * loops + blk + seg * HALF_EDGES;
		ulong live = one_bit_set[nonce0 >> 6];
    const ulong last = dipblock(sipkeys, nonce0, buf);
    for (uint e = 0; e < EDGE_BLOCK_SIZE; e++) {
			int row = -1, counter = -1;
      if((live & ((ulong)1 << e)) != 0) {
     	 	ulong edge = buf[e] ^ last;
      	uint node0 = edge & EDGEMASK;
				uint node1 = (edge >> 32) & EDGEMASK;
				uint node = (round & 1) ? node1 : node0;
				row = node >> YZBITS;
				counter = min((int)atomic_add(counters + row, 1), (int)(FLUSHA2-1)); // assuming ROWS_LIMIT_LOSSES checked
				tmp[row][counter] = make_uint2(node, nonce0 + e);
			}
      barrier(CLK_LOCAL_MEM_FENCE);
      if (counter == FLUSHA-1) {
        int localIdx = min(FLUSHA2, counters[row]);
        int newCount = localIdx % FLUSHA;
        int nflush = localIdx - newCount;
				uint grp = row*NY + col;
        int cnt = min((int)atomic_add(indexes + grp, nflush), (int)(maxOut - nflush));
				__global uint2*buffer = buffer1;
				uint grp1 = grp;
				if(row >= 32){
					buffer = buffer2;
					grp1 = (row-32)*NY + col;
				}
        for (int i = 0; i < nflush; i += 1){
					buffer[((ulong)grp1 * maxOut + cnt + i)] = tmp[row][i];
				}
				
        for (int t = 0; t < newCount; t++) {
          tmp[row][t] = tmp[row][t + nflush];
        }
        counters[row] = newCount;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
	}
  uint2 zero = make_uint2(0, 0);
	for (int row = lid; row < NX; row += dim) {
		int localIdx = min(FLUSHA2, counters[row]);
		uint grp = row * NY + col;
		__global uint2*buffer = buffer1;
		uint grp1 = grp;
		if(row >= 32){
			buffer = buffer2;
			grp1 = (row-32)*NY + col;
		}

		for (int j = localIdx; j % TMPPERLL4; j++)
			tmp[row][j] = zero;

		if(localIdx > 0){
			int cnt = min((int)atomic_add(indexes + grp, localIdx), (int)(maxOut - localIdx));
			for (int i = 0; i < localIdx; i += 1) {
							buffer[((ulong)grp1 * maxOut + cnt + i)] = tmp[row][i];
			}
		}
	}
}
__kernel void 
Cuckaroo_SeedA(const int round, const int seg,
		__constant const siphash_keys* sipkeys,
		__global uint2 * __restrict__ buffer1,
		__global uint2 * __restrict__ buffer2,
		__global uint * __restrict__ indexes, 
		const int maxOut,
		__global ulong *one_bit_set) {
  const int group = get_group_id(0);//blockIdx.x;
  const int dim = get_local_size(0);//blockDim.x;
  const int lid = get_local_id(0);//threadIdx.x;
  const int gid = group * dim + lid;
  const int nthreads = get_global_size(0);//gridDim.x * dim;
  //const int FLUSHA2 = 2*FLUSHA;
  
  __local uint2 tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned
  const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
  __local int counters[NX];
  ulong buf[EDGE_BLOCK_SIZE];

  for (int row = lid; row < NX; row += dim)
    counters[row] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

//  const uint tmp_offset = offset / sizeof(uint2);
  const int col = group % NY;
  const int loops = NEDGES / nthreads; // assuming THREADS_HAVE_EDGES checked
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
   uint nonce0 = gid * loops + blk;
		ulong bit_v = one_bit_set[nonce0 >> 6];
    const ulong last = dipblock(sipkeys, nonce0, buf);
    for (uint e = 0; e < EDGE_BLOCK_SIZE; e++) {
			ulong live = bit_v & ((ulong)1 << e);
			int counter = -1, row = -1;
			if(live > 0){
      	ulong edge = buf[e] ^ last;
				uint node0 = edge & EDGEMASK;
				uint node1 = (edge >> 32) & EDGEMASK;
				row = node0 >> YZBITS;
				counter = min((int)atomic_add(counters + row, 1), (int)(FLUSHA2-1)); 
				tmp[row][counter] = make_uint2(node0, node1);
			}
      barrier(CLK_LOCAL_MEM_FENCE);
      if (counter == FLUSHA-1) {
        int localIdx = min(FLUSHA2, counters[row]);
        int newCount = localIdx % FLUSHA;
        int nflush = localIdx - newCount;
				uint grp = row*NY + col;
				__global uint2*buffer = buffer1;
				uint grp1 = grp;
				if(row >= 32){
								buffer = buffer2;
								grp1 = (row-32)*NY + col;
				}
        int cnt = min((int)atomic_add(indexes + grp, nflush), (int)(maxOut - nflush));
        for (int i = 0; i < nflush; i += 1){
								buffer[((ulong)grp1 * maxOut + cnt + i)] = tmp[row][i];
		}
        for (int t = 0; t < newCount; t++) {
          tmp[row][t] = tmp[row][t + nflush];
        }
        counters[row] = newCount;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  uint2 zero = make_uint2(0, 0);
  for (int row = lid; row < NX; row += dim) {
    int localIdx = min(FLUSHA2, counters[row]);
    uint grp = row * NY + col;
		__global uint2*buffer = buffer1;
		uint grp1 = grp;
		if(row >= 32){
			buffer = buffer2;
			grp1 = (row-32)*NY + col;
		}
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[row][j] = zero;
    
		int cnt = min((int)atomic_add(indexes + grp, localIdx), (int)(maxOut - localIdx));
    for (int i = 0; i < localIdx; i += 1) {
						buffer[((ulong)grp1 * maxOut + cnt + i)] = tmp[row][i];
    }
  }
}


__kernel void
SeedB(__constant const siphash_keys * sipkeys,
		__global uint2 * __restrict__ src,
		__global uint2 * __restrict__ dst,
		__global const uint *__restrict__ srcIdx, 
		__global uint *__restrict__ dstIdx, 
		const int maxOut, const int dstIdx_offset)
{
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    const int lid = get_local_id(0);	//threadIdx.x;
    const int gid = get_global_id(0);

    __local uint2 tmp[NY][FLUSHB2];

		const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
    __local int counters[NY];

    for (int col = lid; col < NY; col += dim)
						counters[col] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int row = group / NY;
    const int bucketEdges = min((int) srcIdx[group], (int) maxOut);
    const int loops = (bucketEdges + dim - 1) / dim;

    for (int loop = 0; loop < loops; loop++){
			int col;
			int counter = 0;
			const int edgeIndex = loop * dim + lid;

			if (edgeIndex < bucketEdges)
			{
				const int index = group * maxOut + edgeIndex;
				uint2 edge = src[index];
				if (!null2(edge)){ 
				uint node1 = edge.x;//endpoint2(sipkeys, edge, 0);
				col = (node1 >> ZBITS) & YMASK;
				counter = min((int) atomic_add(counters + col, 1), (int) (FLUSHB2 - 1));
				tmp[col][counter] = edge;
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			if (counter == FLUSHB - 1)
			{
				int localIdx = min(FLUSHB2, counters[col]);
				int newCount = localIdx % FLUSHB;
				int nflush = localIdx - newCount;
				int cnt = min((int) atomic_add(dstIdx + dstIdx_offset + row * NY + col,
																nflush), (int) (maxOut - nflush));
				for (int i = 0; i < nflush; i += 1)
				{
					dst[((ulong) (row * NY + col) * maxOut + cnt + i)] = tmp[col][i];
				}
				for (int t = 0; t < newCount; t++)
				{
					tmp[col][t] = tmp[col][t + nflush];
				}
				counters[col] = newCount;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		uint2 zero = make_uint2(0,0);
    for (int col = lid; col < NY; col += dim)
    {
			int localIdx = min(FLUSHB2, counters[col]);
			for(int j = localIdx; j % TMPPERLL4; j++)
							tmp[col][j] = zero;

			int cnt = min((int) atomic_add(dstIdx + dstIdx_offset + row * NY + col, localIdx), (int) (maxOut - localIdx));
			for (int i = 0; i < localIdx; i += 1)
			{
							dst[((ulong) (row * NY + col) * maxOut + cnt + i)] = tmp[col][i];//*(ulong4*)&t;
			}
    }
}

__kernel void Mark(__global uint2 *src,
		__global uint *srcIdx, 
		__global uint *two_bit_set,
		const uint maxIn, const uint set_offset){
  const int group = get_group_id(0) ;//blockIdx.x;
  const int dim = get_local_size(0);//blockDim.x;
  const int lid = get_local_id(0);//threadIdx.x;

  __local uint ecounters[COUNTERWORDS];

  int start_index = (group+set_offset) * NZ/16;
  for (int i = lid; i < COUNTERWORDS; i += dim)
    ecounters[i] = two_bit_set[start_index + i];
	barrier(CLK_LOCAL_MEM_FENCE);

  const int edgesInBucket = min(srcIdx[group], maxIn);
  // if (!group && !lid) printf("round %d size  %d\n", round, edgesInBucket);
  const int loops = (edgesInBucket + dim-1) / dim;

  for (int loop = 0; loop < loops; loop++) {
      const int lindex = loop * dim + lid;
      if (lindex < edgesInBucket) {
          const int index = maxIn * group + lindex;
          uint2 edge = src[index];
          if (null2(edge)) continue;
          uint node = edge.x;
          Increase2bCounter(ecounters, node & ZMASK);
      }
  }

	barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = lid; i < COUNTERWORDS; i += dim){
      two_bit_set[start_index + i] = ecounters[i];
  }
}

__kernel void Round1(const int round, __global uint2 * src, 
	__global uint * srcIdx, __global uint* two_bit_set, 
	__global ulong* one_bit_set, const uint maxIn, const uint set_offset) {
  const int group = get_group_id(0);//blockIdx.x;
  const int dim = get_local_size(0);//blockDim.x;
  const int lid = get_local_id(0);//threadIdx.x;

  __local uint ecounters[COUNTERWORDS];

  int start_index = (group+set_offset) * NZ/16 ;
  for (int i = lid; i < COUNTERWORDS; i += dim)
    ecounters[i] = two_bit_set[start_index + i];
	barrier(CLK_LOCAL_MEM_FENCE);

  const int edgesInBucket = min(srcIdx[group], maxIn);
  const int loops = (edgesInBucket + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
      const int lindex = loop * dim + lid;
      if (lindex < edgesInBucket) {
          const int index = maxIn * group + lindex;
          uint2 edge = src[index];
          if (null2(edge)) continue;
          uint node0 = edge.x;
          if (0 == Read2bCounter(ecounters, node0 & ZMASK)) {
              uint nonce = edge.y;
              uint bit_set_index1 = nonce >> 6;
              uint bit_set_index2 = nonce & 63;
              ulong bit_set_v = ~((ulong)1 << bit_set_index2);
              one_bit_set[bit_set_index1] &= bit_set_v;
          }
      }
  }
}

#ifndef PART_BITS
// #bits used to partition edge set processing to save shared memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

#define PART_MASK  ((1 << PART_BITS) - 1) // 1
#define NONPART_BITS (ZBITS - PART_BITS) // ZBITS
#define NONPART_MASK ((1 << NONPART_BITS) - 1) // 1 << ZBITS
#define BITMAPBYTES ((NZ >> PART_BITS) / 16) // NZ / 8

__kernel void
Round(const int round, const int part,
			__constant const siphash_keys * sipkeys,
      __global const uint2* __restrict__ src,
      __global uint2* __restrict__ dst,
	  __global const uint *__restrict__ srcIdx, 
	  __global uint *__restrict__ dstIdx, 
	  const uint maxIn0, const uint maxIn, const uint maxOut) 
{
	const int group = get_group_id(0);	//blockIdx.x;
	const int dim = get_local_size(0);	//blockDim.x;
	const int lid = get_local_id(0);	//threadIdx.x;
	__local uint ebitmap[BITMAPBYTES];
	for (int i = lid; i < BITMAPBYTES; i += dim)
		ebitmap[i] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	int edgesInBucket = min(srcIdx[group], maxIn);
	int loops = (edgesInBucket + dim - 1) / dim;
	for (int loop = 0; loop < loops; loop++)
	{
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint2 edge = src[index];
			if (null2(edge))		continue;
			uint z = endpoint2(sipkeys, edge, round & 1) & ZMASK;
			if((z >> NONPART_BITS) == part){
				bitmapset(ebitmap, z & NONPART_MASK);
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int loop = 0; loop < loops; loop++)
	{
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint2 edge = src[index];
			if (null2(edge))	continue;
			uint node0 = endpoint2(sipkeys, edge, (round & 1));
			uint z = node0 & ZMASK;
			if((z >> NONPART_BITS) == part && bitmaptest(ebitmap, z))
			{
				uint node1 = endpoint2(sipkeys, edge, (round & 1)^1);
				int bucket = node1 >> ZBITS;
				const int bktIdx = atomic_add(dstIdx + bucket, 1);//, maxOut - 1);
				if(bktIdx < maxOut){
					dst[bucket * maxOut + bktIdx] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
				}
			}
    }
  }
}
__kernel void Round2(__global const uint2* __restrict src, __global uint2* dst, 
										__global const uint* srcIdx, __global uint* dstIdx, __global const uint* two_bit_set,
										const uint maxIn, const uint maxOut){
	const int group = get_group_id(0);
	const int dim = get_local_size(0);	//blockDim.x;
	const int lid = get_local_id(0);	//threadIdx.x;
	__local uint ebitmap[BITMAPBYTES];
	for (int i = lid; i < BITMAPBYTES; i += dim)
		ebitmap[i] = two_bit_set[group*NZ/16+i];
	barrier(CLK_LOCAL_MEM_FENCE);

	int edgesInBucket = min(srcIdx[group], maxIn);
	int loops = (edgesInBucket + dim - 1) / dim;
	for (int loop = 0; loop < loops; loop++)
	{
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint2 edge = src[index];
			if (null2(edge))	continue;
			uint node0 = edge.y;//endpoint2(sipkeys, edge, round & 1);
			uint z = node0 & ZMASK;
			if(bitmaptest(ebitmap, z))
			{
				uint node1 = edge.x;//endpoint2(sipkeys, edge, (round & 1) ^ 1);
				int bucket = node1 >> ZBITS;
				const int bktIdx = atomic_add(dstIdx + bucket, 1);//, maxOut - 1);
				if(bktIdx < maxOut){
					dst[bucket * maxOut + bktIdx] = edge;
				}
			}
    }
  }
}
__kernel void Round3(__global const uint2* __restrict src1, 
										__global const uint2 * src2,
										__global uint2* dst, 
										__global const uint* srcIdx1, 
										__global const uint* srcIdx2, 
										__global uint* dstIdx,
										const uint maxIn, const uint maxOut){
	const int group = get_group_id(0);
	const int dim = get_local_size(0);	//blockDim.x;
	const int lid = get_local_id(0);	//threadIdx.x;
	__local uint ebitmap[BITMAPBYTES];
	for (int i = lid; i < BITMAPBYTES; i += dim)
		ebitmap[i] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	int edgesInBucket = min(srcIdx1[group], maxIn);
	int loops = (edgesInBucket + dim-1)/dim;
	for(int loop = 0; loop < loops; ++loop){
		const int lindex = loop * dim + lid;
		if(lindex < edgesInBucket){
			const int index = maxIn * group + lindex;
			uint2 edge = src1[index];
			if(null2(edge)) continue;
			uint node0 = edge.y;
			uint z = node0 & ZMASK;
			bitmapset(ebitmap, z);
		}
	}
	edgesInBucket = min(srcIdx2[group], maxIn);
	loops = (edgesInBucket + dim-1)/dim;
	for(int loop = 0; loop < loops; ++loop){
		const int lindex = loop * dim + lid;
		if(lindex < edgesInBucket){
			const int index = maxIn * group + lindex;
			uint2 edge = src2[index];
			if(null2(edge)) continue;
			uint node0 = edge.y;
			uint z = node0 & ZMASK;
			bitmapset(ebitmap, z);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	
	edgesInBucket = min(srcIdx1[group], maxIn);
	loops = (edgesInBucket + dim-1)/dim;
	for(int loop = 0; loop < loops; ++loop){
		const int lindex = loop * dim + lid;
		if(lindex < edgesInBucket){
			const int index = maxIn * group + lindex;
			uint2 edge = src1[index];
			if(null2(edge)) continue;
			uint node0 = edge.y;
			uint z = node0 & ZMASK;
			if(bitmaptest(ebitmap, z)){
				uint node1 = edge.x;
				const int bucket = node1 >> ZBITS;
				const int bktIdx = atomic_add(dstIdx + bucket, 1);
				if(bktIdx < maxOut){
					dst[bucket * maxOut + bktIdx] = edge;
				}
			}
		}
	}
	edgesInBucket = min(srcIdx2[group], maxIn);
	loops = (edgesInBucket + dim-1)/dim;
	for(int loop = 0; loop < loops; ++loop){
		const int lindex = loop * dim + lid;
		if(lindex < edgesInBucket){
			const int index = maxIn * group + lindex;
			uint2 edge = src2[index];
			if(null2(edge)) continue;
			uint node0 = edge.y;
			uint z = node0 & ZMASK;
			if(bitmaptest(ebitmap, z)){
				uint node1 = edge.x;
				const int bucket = node1 >> ZBITS;
				const int bktIdx = atomic_add(dstIdx + bucket, 1);
				if(bktIdx < maxOut){
					dst[bucket * maxOut + bktIdx] = edge;
				}
			}
		}
	}
}

__kernel void
Tail(__global const uint2 * source, __global uchar * destination, __global const uint *srcIdx, __global uint *dstIdx, const int maxIn)
{
	__global uint2* dst = (__global uint2*)(destination);
    const int lid = get_local_id(0);	//threadIdx.x;
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    int myEdges = srcIdx[group];
    __local int destIdx;

    if (lid == 0)
	destIdx = atomic_add(dstIdx, myEdges);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < myEdges; i += dim)
	dst[destIdx + i] = source[group * maxIn + i];
}

)";

inline std::string get_kernel_source ()
{
	return kernel_source;
}
#endif
