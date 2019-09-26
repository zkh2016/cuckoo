// Cuckaroo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2019 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license

#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "cuckaroo.hpp"
#include "graph.hpp"
//#include "../crypto/siphash.cuh"
#include "../crypto/blake2.h"
#include "ocl.h"
#include "kernel_source.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint64_t u64; // save some typing

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

#ifndef IDXSHIFT
// number of bits of compression of surviving edge endpoints
// reduces space used in cycle finding, but too high a value
// results in NODE OVERFLOW warnings and fake cycles
#define IDXSHIFT 12
#endif

const u32 MAXEDGES = NEDGES >> IDXSHIFT;

#ifndef XBITS
#define XBITS 6
#endif
#define YBITS 7

#define NODEBITS (EDGEBITS + 1)

const u32 NX        = 1 << XBITS;
const u32 NY        = 1 << YBITS;
const u32 NX2       = NX * NY;
const u32 NXY       = NX * NY;
const u32 XMASK     = NX - 1;
const u32 YMASK     = NY - 1;
//const u32 YBITS     = XBITS;
const u32 YZBITS    = EDGEBITS - XBITS;
const u32 ZBITS     = YZBITS - YBITS;
const u32 NZ        = 1 << ZBITS;
const u32 ZMASK     = NZ - 1;

#ifndef NEPS_A
#define NEPS_A 158
#endif
#ifndef NEPS_B
#define NEPS_B 92
#endif
#define NEPS 128

#define SEG 2
const u32 EDGES_A = NZ * NEPS_A / SEG / NEPS;
const u32 EDGES_B = NZ * NEPS_B / SEG / NEPS;

const u32 ROW_EDGES_A = EDGES_A * NY;
const u32 ROW_EDGES_B = EDGES_B * NY;

#ifndef NRB1
#define NRB1 (NX / 2)
#endif
#define NRB2 NRB1
// Number of Parts of BufferB, all but one of which will overlap BufferA
#ifndef NB
#define NB 2
#endif

#ifndef NA
#define NA 4//((NB * NEPS_A + NEPS_B-1) / NEPS_B)
#endif

const u32 ZERO = 0;

#ifndef PART_BITS
// #bits used to partition edge set processing to save shared memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

const u32 PART_MASK = (1 << PART_BITS) - 1; // 1
const u32 NONPART_BITS = ZBITS - PART_BITS; // ZBITS
const word_t NONPART_MASK = (1 << NONPART_BITS) - 1; // 1 << ZBITS
const int BITMAPBYTES = (NZ >> PART_BITS) / 4; // NZ / 8

#define checkOpenclErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cl_int code, const char *file, int line, bool abort=true) {
  if (code != CL_SUCCESS) {
    fprintf(stderr, "GPUassert: %s %s %d\n", openclGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

struct blockstpb {
  u16 blocks;
  u16 tpb;
};

struct trimparams {
  u16 ntrims;
  blockstpb genA;
  blockstpb genB;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;

  trimparams() {
    ntrims              =  90;
    genA.blocks         = NX2;
    genA.tpb            =  256;
    genB.blocks         =  NX2;
    genB.tpb            =  128;
    trim.blocks         =  NX2;
    trim.tpb            =  256;
    tail.blocks         =  NX2;
    tail.tpb            = 1024;
    recover.blocks      = 1024;
    recover.tpb         = 1024;
  }
};

typedef u32 proof[PROOFSIZE];

void get_surviving(cl_command_queue commandQueue, cl_mem one_bit_set){
	return;
	u64 *lives = new u64[NX2*NZ/64];
	cl_int clResult = clEnqueueReadBuffer(commandQueue, one_bit_set, CL_TRUE, 0, NX2*NZ/8, lives, 0, NULL, NULL); 
	checkOpenclErrors(clResult);
	clFinish(commandQueue);
	u64 surviving = 0;
	for(int i = 0; i < NX2*NZ/64; i++){
		surviving += __builtin_popcountll(lives[i]);
	}
	printf("surving = %lu\n", surviving);
	delete lives;
}
// maintains set of trimmable edges
struct edgetrimmer {
  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_context context;
  cl_command_queue commandQueue;
  cl_program program;
  trimparams tp;
  edgetrimmer *dt;
  size_t sizeA, sizeB;
  const size_t indexesSize = NX * NY * sizeof(u32);
  cl_mem bufferA;
  cl_mem bufferB;
  cl_mem bufferC;
  cl_mem indexesE;
  cl_mem indexesE1;
  cl_mem indexesE2;
  cl_mem recoveredges; //const
  u32 nedges;
	cl_mem uvnodes;
  siphash_keys sipkeys;
  cl_mem dipkeys;
	cl_mem dipkeys2;
  bool abort;
  bool initsuccess = false;
	cl_mem one_bit_set;
	cl_mem two_bit_set;

  edgetrimmer(const trimparams _tp) : tp(_tp) {
  	platformId = getOnePlatform ();
		if (platformId == NULL){
						printf("get null platform..\n");
						return;
		}
		//	getPlatformInfo (platformId);
		deviceId = getOneDevice (platformId, 1);
		if (deviceId == NULL){
						printf("get null device..\n");
						exit(0);
		}
		context = createContext (platformId, deviceId);
		if (context == NULL){
						printf("get null context..\n");
						return;
		}
		commandQueue = createCommandQueue (context, deviceId);
		if (commandQueue == NULL){
						printf("get null command queue..\n");
						return ;
		}

		string sourceStr = get_kernel_source();
		size_t size = sourceStr.size();
		const char *source = sourceStr.c_str ();
		program = createProgram (context, &source, size);

		//	cl_program program = createByBinaryFile("trimmer.bin", context, deviceId);	
		if (program == NULL){
						printf("create program error\n");
						return ;
		}
		printf("EDGEBITS = %d, PROOFSIZE = %d\n", EDGEBITS, PROOFSIZE);
		char options[1024] = "-I./";
		sprintf (options, "-I./ -DEDGEBITS=%d -DPROOFSIZE=%d ", EDGEBITS, PROOFSIZE);

		buildProgram (program, &(deviceId), options);
		initsuccess = true;

		tp = _tp;

		cl_int clResult;
		dipkeys = clCreateBuffer(this->context, CL_MEM_READ_ONLY, sizeof (siphash_keys), NULL, &clResult);
		checkOpenclErrors(clResult);
		dipkeys2 = clCreateBuffer(this->context, CL_MEM_READ_ONLY, sizeof (siphash_keys), NULL, &clResult);
		checkOpenclErrors(clResult);

		indexesE = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize, NULL, &clResult);
		indexesE1 = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize, NULL, &clResult);
		indexesE2 = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize, NULL, &clResult);
		checkOpenclErrors(clResult);

		recoveredges = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof (cl_uint2) * PROOFSIZE, NULL, &clResult);
		checkOpenclErrors(clResult);

		sizeA = ROW_EDGES_A * NX * sizeof (cl_uint2);
		sizeB = ROW_EDGES_B * NX * sizeof (cl_uint2);

		const size_t nonoverlap = sizeB * NRB1 / NX;
		const size_t bufferSize = sizeA/2;
		//	fprintf(stderr, "bufferSize: %lu\n", bufferSize);
		bufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, &clResult);
		checkOpenclErrors(clResult);
		bufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, &clResult);
		checkOpenclErrors(clResult);
		bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, &clResult);
		checkOpenclErrors(clResult);

		uvnodes = clCreateBuffer(context, CL_MEM_READ_WRITE, indexesSize, NULL, &clResult);
		checkOpenclErrors(clResult);
		one_bit_set = clCreateBuffer(context, CL_MEM_READ_WRITE, NX2*NZ/8, NULL, &clResult);
		checkOpenclErrors(clResult);
		two_bit_set = clCreateBuffer(context, CL_MEM_READ_WRITE, NX2*NZ/4, NULL, &clResult);
		checkOpenclErrors(clResult);
  }
  u64 globalbytes() const {
    return (sizeA+sizeA/2) + (1+NB) * indexesSize + sizeof(siphash_keys) + PROOFSIZE * 2 * sizeof(u32) + sizeof(edgetrimmer) + NX2*NZ/8 + NX2*NZ/4;
  }
  ~edgetrimmer() {
  }

void save_to_file(cl_mem mem, size_t offset1, cl_mem indexes, size_t offset2){
	u32 *tindexs = new u32[NX2*NZ];
	cl_uint2 *tmpbuf = new cl_uint2[sizeA];
	clEnqueueReadBuffer(commandQueue, indexes, CL_TRUE, offset2, indexesSize, tindexs, 0, NULL, NULL); 	
	FILE *fp = fopen("cl.txt", "w");
	for(int i = 0; i < NX2 * NZ; i++){
		//fprintf(fp, "%u\n", tindexs[i]);
		size_t size1 = tindexs[i];
		size_t size = (size1 > EDGES_A ? EDGES_A : size1) * sizeof(cl_uint2);
		clEnqueueReadBuffer(commandQueue, mem, CL_TRUE, offset1 + EDGES_A * i * sizeof(cl_uint2), size, tmpbuf, 0, NULL, NULL); 
		for(int j = 0; j < size/sizeof(cl_uint2); j++){
			if(tmpbuf[j].y != 0 || tmpbuf[j].x != 0)
			fprintf(fp, "%u\n", tmpbuf[j].x);
		}
	}
	fclose(fp);
	delete tindexs;
	delete tmpbuf;
}
  void check_repeat3(cl_mem dindexes, cl_mem dbuffer1,  u32 maxOut){
		printf("check repeat....\n");
		u32 *hindexes = new u32[NX*NY];
		int index = 0;
		size_t offset1 = 0, offset2=0, offset = 0;
		cl_uint2 *hbuffer = new cl_uint2[sizeA];
		cl_mem di[1] = {dindexes};
		cl_mem db[1] = {dbuffer1};
		for(int n = 0; n < 1; n++){
		clEnqueueReadBuffer(commandQueue, di[n], CL_TRUE, 0, indexesSize, hindexes, 0, NULL, NULL); 
			offset = 0;
			for(int i = 0; i < NX*NY; i++){
				int size = maxOut < hindexes[i] ? maxOut : hindexes[i];
				clEnqueueReadBuffer(commandQueue, db[n], CL_TRUE, offset *sizeof(cl_uint2), size*sizeof(cl_uint2), hbuffer + index, 0, NULL, NULL);
				index += size;
				offset += maxOut;
			}
		}

		std::sort(hbuffer, hbuffer+index, [](const cl_uint2 a, const cl_uint2 b)-> bool {
				if(a.x == b.x) return a.y > b.y;
				else return a.x > b.x;
		});
		
		bool flag = false;
		for(int i = 1; i < index; i++){
			cl_uint2 a = hbuffer[i-1];
			cl_uint2 b = hbuffer[i];
			if(a.x == b.x && a.y == b.y && a.x != 0 && a.y != 0){
				printf("find same edges....(%u,%u), (%u,%u)\n", a.x, a.y, b.x, b.y);
				flag = true;
			}
		}
		if(flag) exit(0);
  }
  void check_repeat2(cl_mem dindexes, cl_mem dindexes2, cl_mem dbuffer1, cl_mem dbuffer2, u32 maxOut){
		printf("check repeat....\n");
		u32 *hindexes = new u32[NX*NY];
		int index = 0;
		size_t offset1 = 0, offset2=0, offset = 0;
		cl_uint2 *hbuffer = new cl_uint2[sizeA];
		cl_mem di[2] = {dindexes, dindexes2};
		cl_mem db[2] = {dbuffer1, dbuffer2};
		for(int n = 0; n < 2; n++){
		clEnqueueReadBuffer(commandQueue, di[n], CL_TRUE, 0, indexesSize, hindexes, 0, NULL, NULL); 
			offset = 0;
			for(int i = 0; i < NX*NY; i++){
				int size = maxOut < hindexes[i] ? maxOut : hindexes[i];
				clEnqueueReadBuffer(commandQueue, db[n], CL_TRUE, offset *sizeof(cl_uint2), size*sizeof(cl_uint2), hbuffer + index, 0, NULL, NULL);
				index += size;
				offset += maxOut;
			}
		}

		std::sort(hbuffer, hbuffer+index, [](const cl_uint2 a, const cl_uint2 b)-> bool {
				if(a.x == b.x) return a.y > b.y;
				else return a.x > b.x;
		});
		
		bool flag = false;
		for(int i = 1; i < index; i++){
			cl_uint2 a = hbuffer[i-1];
			cl_uint2 b = hbuffer[i];
			if(a.x == b.x && a.y == b.y && a.x != 0 && a.y != 0){
				printf("find same edges....(%u,%u), (%u,%u)\n", a.x, a.y, b.x, b.y);
				flag = true;
			}
		}
		if(flag) exit(0);
  }
  void check_repeat(cl_mem dindexes, cl_mem dbuffer1, cl_mem dbuffer2, u32 maxOut){
		printf("check repeat....\n");
		u32 *hindexes = new u32[NX*NY];
		int index = 0;
		size_t offset1 = 0, offset2=0, offset = 0;
		cl_uint2 *hbuffer = new cl_uint2[sizeA];
		clEnqueueReadBuffer(commandQueue, dindexes, CL_TRUE, 0, indexesSize, hindexes, 0, NULL, NULL); 
			for(int i = 0; i < NX*NY; i++){
				int size = maxOut < hindexes[i] ? maxOut : hindexes[i];
				cl_mem mem = dbuffer1;
				offset = offset1;
				if(i >= NX*NX){
					mem = dbuffer2;
					offset = offset2;
				}	
				clEnqueueReadBuffer(commandQueue, mem, CL_TRUE, offset *sizeof(cl_uint2), size*sizeof(cl_uint2), hbuffer + index, 0, NULL, NULL);
				index += size;
				if(i >= NX*NX){
					offset2 += maxOut;
				}else{ offset1 += maxOut; }
			}

		std::sort(hbuffer, hbuffer+index, [](const cl_uint2 a, const cl_uint2 b)-> bool {
				if(a.x == b.x) return a.y > b.y;
				else return a.x > b.x;
		});
		
		bool flag = false;
		for(int i = 1; i < index; i++){
			cl_uint2 a = hbuffer[i-1];
			cl_uint2 b = hbuffer[i];
			if(a.x == b.x && a.y == b.y && a.x != 0 && a.y != 0){
				printf("find same edges....(%u,%u), (%u,%u)\n", a.x, a.y, b.x, b.y);
				flag = true;
			}
		}
		if(flag) exit(0);
  }
	void count_edges(cl_mem indexesE, u32 offset, u32 maxIn){
return;
		u32 *tmpindex = new u32[NX2];
   // cudaMemcpy(tmpindex, indexesE, NX*NY*sizeof(u32), cudaMemcpyDeviceToHost);
		clEnqueueReadBuffer(commandQueue, indexesE, CL_TRUE, offset, indexesSize, tmpindex, 0, NULL, NULL);
    u32 sum = 0;
    for(int i = 0; i < NX*NY; i++){
        sum += tmpindex[i] > maxIn ? maxIn : tmpindex[i];
    }
    printf("edges count : %llu\n", sum);
		delete tmpindex;
	}
void test(cl_mem two_bit_set){
	u32* bitset = new u32[NX2*NZ/4/sizeof(u32)];
	clEnqueueReadBuffer(commandQueue, two_bit_set, CL_TRUE, 0, NX2*NZ/4, bitset, 0, NULL, NULL);
	//FILE *fp = fopen("cl_two_bit_set.txt", "w");
	u32 sum = 0;
	for(int i = 0; i < NX2; i++){
		for(int j = 0; j < NZ/32; j++){
			u32 tmp = bitset[i*NZ/16 + NZ/32+j];
			for(int k = 31; k >= 0; k--){
//				fprintf(fp, "%u\n", (tmp>>k) & 1);
				sum += ((tmp>>k)&1);
			}
		}
	}  
//	fclose(fp);
	printf("live edge %u...\n", sum);
	delete bitset;
}
void SeedA(cl_kernel seedA_kernel, u32 round, u32 i){
	cl_int clResult;
  u32 bufAB_offset = sizeB  * NRB1 / NX;
  uint srcIdx_offset = indexesSize / sizeof(u32);
  int edges_a = EDGES_A;
	size_t global_work_size[1];
	size_t local_work_size[1];
  global_work_size[0] = tp.genA.blocks * tp.genA.tpb;
  local_work_size[0] = tp.genA.tpb;
	clResult = clEnqueueFillBuffer(commandQueue, indexesE, &ZERO, sizeof(int), 0, indexesSize, 0, NULL, NULL); 
	checkOpenclErrors(clResult);

  clResult = clSetKernelArg(seedA_kernel, 0, sizeof (int), &round);
  clResult |= clSetKernelArg(seedA_kernel, 1, sizeof (int), &i);
  clResult |= clSetKernelArg(seedA_kernel, 2, sizeof (cl_mem), (void *) &dipkeys);
  clResult |= clSetKernelArg(seedA_kernel, 3, sizeof (cl_mem),(void *) &bufferA);
  clResult |= clSetKernelArg(seedA_kernel, 4, sizeof (cl_mem),(void *) &bufferB);
  clResult |= clSetKernelArg(seedA_kernel, 5, sizeof (cl_mem), (void *) &indexesE);
  clResult |= clSetKernelArg(seedA_kernel, 6, sizeof (int), &edges_a);
  clResult |= clSetKernelArg(seedA_kernel, 7, sizeof (cl_mem), (void*)&one_bit_set);
  checkOpenclErrors(clResult);
  clResult |= clEnqueueNDRangeKernel(commandQueue, seedA_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  checkOpenclErrors(clResult);
  clFinish(commandQueue);
//	check_repeat(indexesE, bufferA, bufferB, EDGES_A);
	//save_to_file(bufferA, bufAB_offset, indexesE, indexesSize);
}

void SeedB(){
	cl_int clResult;
	size_t tmpSize = indexesSize;
  int edges_a = EDGES_A;

  size_t global_work_size[1];
  size_t local_work_size[1];

  cl_kernel seedB_kernel = clCreateKernel(program, "SeedB", &clResult);
  global_work_size[0] = tp.genB.blocks / 2 * tp.genB.tpb;
  local_work_size[0] = tp.genB.tpb;

	clResult = clEnqueueFillBuffer(commandQueue, indexesE1, &ZERO, sizeof(int), 0, indexesSize, 0, NULL, NULL); 
	clResult = clEnqueueCopyBuffer(commandQueue, indexesE, indexesE2, indexesSize/2, 0, indexesSize/2, 0, NULL, NULL);
	checkOpenclErrors(clResult);
	clFinish(commandQueue);

	cl_mem mem[2], mem2[2], mem3[3];
	mem[0] = bufferA; mem[1] = bufferB;
	mem2[0] = bufferC; mem2[1] = bufferA;
	mem3[0] = indexesE; mem3[1] = indexesE2;
	int dstIdx_offset[2] = {0, NX2/2};
  for(u32 i = 0; i < 2; i++){
     clResult |= clSetKernelArg(seedB_kernel, 0, sizeof (cl_mem), (void *) &dipkeys);
     clResult |= clSetKernelArg(seedB_kernel, 1, sizeof (cl_mem), (void *) &mem[i]);
     clResult |= clSetKernelArg(seedB_kernel, 2, sizeof (cl_mem), (void *) &mem2[i]);
     clResult |= clSetKernelArg(seedB_kernel, 3, sizeof (cl_mem), (void *) &mem3[i]);
     clResult |= clSetKernelArg(seedB_kernel, 4, sizeof (cl_mem), (void *) &indexesE1);
     clResult |= clSetKernelArg(seedB_kernel, 5, sizeof (u32), &edges_a);
     clResult |= clSetKernelArg(seedB_kernel, 6, sizeof (u32), &dstIdx_offset[i]);
     checkOpenclErrors(clResult);

     clResult = clEnqueueNDRangeKernel(commandQueue, seedB_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
     checkOpenclErrors(clResult);
		 clFinish(commandQueue);
  }
	//check_repeat(indexesE, 0, bufferA, 0, EDGES_A);
	//save_to_file(bufferA, 0, indexesE, 0);
}

void Mark(int round){
  u32 maxIn = EDGES_A;
  cl_int clResult;
  cl_kernel mark_kernel = clCreateKernel(program, "Mark", &clResult);
  size_t global_work_size[1];
	global_work_size[0] = tp.trim.blocks/2 * tp.trim.tpb;
  size_t local_work_size[1] = {tp.trim.tpb};
	clResult = clEnqueueCopyBuffer(commandQueue, indexesE1, indexesE, indexesSize/2, 0, indexesSize/2, 0, NULL, NULL);
	cl_mem mem[2] = {bufferC, bufferA};
	cl_mem indexs[2] = {indexesE1, indexesE};
	u32 set_offsets[2] = {0, NX2/2};
	for(int i = 0; i < 2; i++){
  	clResult |= clSetKernelArg(mark_kernel, 0, sizeof(cl_mem), (void*)&mem[i]);
		clResult |= clSetKernelArg(mark_kernel, 1, sizeof(cl_mem), (void*)&indexs[i]);
		clResult |= clSetKernelArg(mark_kernel, 2, sizeof(cl_mem), (void*)&two_bit_set);
		clResult |= clSetKernelArg(mark_kernel, 3, sizeof(u32), &maxIn);
		clResult |= clSetKernelArg(mark_kernel, 4, sizeof(u32), &set_offsets[i]);
		clResult = clEnqueueNDRangeKernel(commandQueue, mark_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		checkOpenclErrors(clResult);
		clFinish(commandQueue);
	}
}

void Mark2(cl_mem buffer, cl_mem indexesE, u32 maxIn){
  cl_int clResult;
  cl_kernel mark_kernel = clCreateKernel(program, "Mark", &clResult);
  size_t global_work_size[1];
	global_work_size[0] = tp.trim.blocks * tp.trim.tpb;
  size_t local_work_size[1] = {tp.trim.tpb};
	u32 round = 1;
	clResult |= clSetKernelArg(mark_kernel, 0, sizeof(cl_mem), (void*)&buffer);
	clResult |= clSetKernelArg(mark_kernel, 1, sizeof(cl_mem), (void*)&indexesE);
	clResult |= clSetKernelArg(mark_kernel, 2, sizeof(cl_mem), (void*)&two_bit_set);
	clResult |= clSetKernelArg(mark_kernel, 3, sizeof(u32), &maxIn);
	clResult |= clSetKernelArg(mark_kernel, 4, sizeof(u32), &round);
	clResult = clEnqueueNDRangeKernel(commandQueue, mark_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	checkOpenclErrors(clResult);
	clFinish(commandQueue);
}
void Round1(u32 round){
  u32 maxIn = EDGES_A;
  cl_int clResult;
  cl_kernel round1_kernel = clCreateKernel(program, "Round1", &clResult);
  size_t global_work_size[1];
	global_work_size[0] = tp.trim.blocks/2 * tp.trim.tpb;
  size_t local_work_size[1] = {tp.trim.tpb};
	cl_mem mem[2] = {bufferC, bufferA};
	cl_mem indexs[2] = {indexesE1, indexesE};
	u32 set_offsets[2] = {0, NX2/2};
	for(int i = 0; i < 2; i++){
  	clResult |= clSetKernelArg(round1_kernel, 0, sizeof(u32), &round);
		clResult |= clSetKernelArg(round1_kernel, 1, sizeof(cl_mem), (void*)&mem[i]); 
		clResult |= clSetKernelArg(round1_kernel, 2, sizeof(cl_mem), (void*)&indexs[i]);
		clResult |= clSetKernelArg(round1_kernel, 3, sizeof(cl_mem), (void*)&two_bit_set);
		clResult |= clSetKernelArg(round1_kernel, 4, sizeof(cl_mem), (void*)&one_bit_set);
		clResult |= clSetKernelArg(round1_kernel, 5, sizeof(u32), &maxIn);
		clResult |= clSetKernelArg(round1_kernel, 6, sizeof(u32), &set_offsets[i]);
		clResult = clEnqueueNDRangeKernel(commandQueue, round1_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		checkOpenclErrors(clResult);
		clFinish(commandQueue);
	}
}
void Round(cl_kernel round1_kernel, cl_mem buffer1, cl_mem buffer2, cl_mem index1, cl_mem index2, 
	const u32 round, const u32 maxIn0, const u32 maxIn, const u32 maxOut,
 	const u32 seg1, const u32 seg2, const u32 part){ 
  cl_int clResult;
  size_t global_work_size[1];
	global_work_size[0] = tp.trim.blocks * seg1 / seg2 * tp.trim.tpb;
  size_t local_work_size[1] = {tp.trim.tpb};
  clResult |= clSetKernelArg(round1_kernel, 0, sizeof(u32), &round);
  clResult |= clSetKernelArg(round1_kernel, 1, sizeof(u32), &part);
  clResult |= clSetKernelArg(round1_kernel, 2, sizeof(cl_mem), (void*)&dipkeys); 
  clResult |= clSetKernelArg(round1_kernel, 3, sizeof(cl_mem), (void*)&buffer1); 
  clResult |= clSetKernelArg(round1_kernel, 4, sizeof(cl_mem), (void*)&buffer2); 
  clResult |= clSetKernelArg(round1_kernel, 5, sizeof(cl_mem), (void*)&index1);
  clResult |= clSetKernelArg(round1_kernel, 6, sizeof(cl_mem), (void*)&index2);
  clResult |= clSetKernelArg(round1_kernel, 7, sizeof(u32), &maxIn0);
  clResult |= clSetKernelArg(round1_kernel, 8, sizeof(u32), &maxIn);
  clResult |= clSetKernelArg(round1_kernel, 9, sizeof(u32), &maxOut);
  clResult = clEnqueueNDRangeKernel(commandQueue, round1_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  checkOpenclErrors(clResult);
  clFinish(commandQueue);
}
void Round2(cl_mem buffer1, cl_mem buffer2, cl_mem index1, cl_mem index2, 
	const u32 maxIn, const u32 maxOut){ 
	cl_int clResult;
	cl_kernel round1_kernel = clCreateKernel(program, "Round2", &clResult);
  size_t global_work_size[1];
	global_work_size[0] = tp.trim.blocks * tp.trim.tpb;
  size_t local_work_size[1] = {tp.trim.tpb};
  clResult |= clSetKernelArg(round1_kernel, 0, sizeof(cl_mem), (void*)&buffer1); 
  clResult |= clSetKernelArg(round1_kernel, 1, sizeof(cl_mem), (void*)&buffer2); 
  clResult |= clSetKernelArg(round1_kernel, 2, sizeof(cl_mem), (void*)&index1);
  clResult |= clSetKernelArg(round1_kernel, 3, sizeof(cl_mem), (void*)&index2);
  clResult |= clSetKernelArg(round1_kernel, 4, sizeof(cl_mem), (void*)&two_bit_set);
  clResult |= clSetKernelArg(round1_kernel, 5, sizeof(u32), &maxIn);
  clResult |= clSetKernelArg(round1_kernel, 6, sizeof(u32), &maxOut);
  clResult = clEnqueueNDRangeKernel(commandQueue, round1_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  checkOpenclErrors(clResult);
  clFinish(commandQueue);
}
void Round3(cl_mem buffer1, cl_mem buffer2, cl_mem buffer3, 
						cl_mem index1, cl_mem index2, cl_mem index3, 
	const u32 maxIn, const u32 maxOut){ 
	cl_int clResult;
	cl_kernel round1_kernel = clCreateKernel(program, "Round3", &clResult);
  size_t global_work_size[1];
	global_work_size[0] = tp.trim.blocks * tp.trim.tpb;
  size_t local_work_size[1] = {tp.trim.tpb};
  clResult |= clSetKernelArg(round1_kernel, 0, sizeof(cl_mem), (void*)&buffer1); 
  clResult |= clSetKernelArg(round1_kernel, 1, sizeof(cl_mem), (void*)&buffer2); 
  clResult |= clSetKernelArg(round1_kernel, 2, sizeof(cl_mem), (void*)&buffer3); 
  clResult |= clSetKernelArg(round1_kernel, 3, sizeof(cl_mem), (void*)&index1);
  clResult |= clSetKernelArg(round1_kernel, 4, sizeof(cl_mem), (void*)&index2);
  clResult |= clSetKernelArg(round1_kernel, 5, sizeof(cl_mem), (void*)&index3);
  clResult |= clSetKernelArg(round1_kernel, 6, sizeof(u32), &maxIn);
  clResult |= clSetKernelArg(round1_kernel, 7, sizeof(u32), &maxOut);
  clResult = clEnqueueNDRangeKernel(commandQueue, round1_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  checkOpenclErrors(clResult);
  clFinish(commandQueue);
}
void Tail(){
	size_t global_work_size[1], local_work_size[1];
	global_work_size[0] = tp.tail.blocks * tp.tail.tpb;
	local_work_size[0] = tp.tail.tpb;
	u32 maxIn = EDGES_A/4;
	cl_int clResult;
	cl_kernel kernel = clCreateKernel(program, "Tail", &clResult);
	clResult |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferA);
	clResult |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferB);
	clResult |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&indexesE);
	clResult |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&indexesE1);
	clResult |= clSetKernelArg(kernel, 4, sizeof(u32), &maxIn);
  clResult = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  checkOpenclErrors(clResult);
  clFinish(commandQueue);
}

  u32 trim() {
		cl_int clResult;
		u64 max64 = 0xFFFFFFFFFFFFFFFF;
		clResult = clEnqueueFillBuffer(commandQueue, one_bit_set, &max64, sizeof (uint64_t), 0, NX2*NZ/8, 0, NULL, NULL);
		checkOpenclErrors(clResult);
		clFinish(commandQueue);
		printf("init....\n");
		get_surviving(commandQueue, one_bit_set);

	printf("%lu %lu %lu %lu\n", sipkeys.k0, sipkeys.k1, sipkeys.k2, sipkeys.k3);
		clResult = clEnqueueWriteBuffer(commandQueue, dipkeys, CL_TRUE, 0, sizeof (siphash_keys), &sipkeys, 0, NULL, NULL);
		checkOpenclErrors(clResult);
		clFinish(commandQueue);

		cl_kernel seedA1_kernel = clCreateKernel(program, "Cuckaroo_SeedA1", &clResult);
		checkOpenclErrors(clResult);
		for(int round = 0; round < 2; ++round){
			clResult = clEnqueueFillBuffer(commandQueue, two_bit_set, &ZERO, sizeof (u32), 0, NX2*NZ/4, 0, NULL, NULL);
			clFinish(commandQueue);
			checkOpenclErrors(clResult);

			for(int i = 0; i < SEG; i++){
				SeedA(seedA1_kernel, round, i);
				SeedB();
				Mark(round);
			}
			Round1(round);
			get_surviving(commandQueue, one_bit_set);
			for(int i = 0; i < SEG-1; i++){
				SeedA(seedA1_kernel, round, i);
				SeedB();
				Round1(round);
				get_surviving(commandQueue, one_bit_set);
			}
		}
		printf("end lean....\n");
/////////////////////////////////////////////////////////
		cl_kernel seedA_kernel = clCreateKernel(program, "Cuckaroo_SeedA", &clResult);
		printf("seeda...\n");
		SeedA(seedA_kernel, 1, 0);
		//check_repeat(indexesE, bufferA, bufferB, EDGES_A);
		//return 0;

		u32 bufAB_offset = sizeB  * NRB1 / NX;
		count_edges(indexesE, 0, EDGES_A); 

		SeedB();
		printf("seedB..\n");
		count_edges(indexesE1, 0, EDGES_A); 
		//check_repeat(indexesE1, bufferC, bufferA, EDGES_A);
		//return 0;

  	cl_kernel round_kernel = clCreateKernel(program, "Round", &clResult);
//  	cl_kernel round2_kernel = clCreateKernel(program, "Round2", &clResult);
	
		printf("round0...\n");
		clResult = clEnqueueFillBuffer(commandQueue, indexesE2, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
		Round(round_kernel, bufferC, bufferB, indexesE1, indexesE2, 0, EDGES_A, EDGES_A, EDGES_A/2, 1, 2, 0);
		clResult = clEnqueueCopyBuffer(commandQueue, indexesE1, indexesE, indexesSize/2, 0, indexesSize/2, 0, NULL, NULL);
		clResult = clEnqueueFillBuffer(commandQueue, indexesE1, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
		Round(round_kernel, bufferA, bufferC, indexesE, indexesE1, 0, EDGES_A, EDGES_A, EDGES_A/2, 1, 2, 0);
		count_edges(indexesE2, 0, EDGES_A/2); 
		count_edges(indexesE1, 0, EDGES_A/2); 
		//check_repeat2(indexesE2, indexesE1, bufferB, bufferC, EDGES_A/2);
		//return 0;

//		clResult = clEnqueueFillBuffer(commandQueue, two_bit_set, &ZERO, sizeof (u32), 0, NX2*NZ/4, 0, NULL, NULL);
//		Mark2(bufferB, indexesE2, EDGES_A/2);
//		//test(two_bit_set);
//		Mark2(bufferC, indexesE1, EDGES_A/2);
		//test(two_bit_set);

		printf("round1...\n");
		clEnqueueFillBuffer(commandQueue, indexesE, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
//		Round2(bufferB, bufferA, indexesE2, indexesE, EDGES_A/2, EDGES_A/2);
//		Round2(bufferC, bufferA, indexesE1, indexesE, EDGES_A/2, EDGES_A/2);
		Round3(bufferB, bufferC, bufferA, indexesE2, indexesE1, indexesE, EDGES_A/2, EDGES_A/2);
		count_edges(indexesE, 0, EDGES_A/2); 
		//check_repeat3(indexesE, bufferA, EDGES_A/2);
		//return 0;

		printf("round2...\n");
		clEnqueueFillBuffer(commandQueue, indexesE1, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
		Round(round_kernel, bufferA, bufferB, indexesE, indexesE1, 2, EDGES_A/2, EDGES_A/2, EDGES_A/4, 1, 1, 0);
		count_edges(indexesE1, 0, EDGES_A/4); 
		//check_repeat3(indexesE1, bufferB, EDGES_A/4);
		//return 0;

		printf("round3...\n");
		clEnqueueFillBuffer(commandQueue, indexesE, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
		Round(round_kernel, bufferB, bufferA, indexesE1, indexesE, 3, EDGES_A/4, EDGES_A/4, EDGES_A/4, 1, 1, 0);
		count_edges(indexesE, 0, EDGES_A/4); 
		//check_repeat3(indexesE, bufferA, EDGES_B/4);
		//return 0;
		for(int round = 4; round < 120; round += 2){
			clEnqueueFillBuffer(commandQueue, indexesE1, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
			Round(round_kernel, bufferA, bufferB, indexesE, indexesE1, round, EDGES_A/4, EDGES_A/4, EDGES_A/4, 1, 1, 0);

			clEnqueueFillBuffer(commandQueue, indexesE, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
			Round(round_kernel, bufferB, bufferA, indexesE1, indexesE, round+1, EDGES_A/4, EDGES_A/4, EDGES_A/4, 1, 1, 0);
		}

		count_edges(indexesE, 0, EDGES_A/4); 
		//check_repeat3(indexesE, bufferA, EDGES_A/4);
		//return 0;
		clEnqueueFillBuffer(commandQueue, indexesE1, &ZERO, sizeof(int), 0, indexesSize,0, NULL, NULL);
		Tail();
		clEnqueueReadBuffer(commandQueue, indexesE1, CL_TRUE, 0, sizeof(u32), &nedges, 0, NULL, NULL);
		printf("nedges = %u\n", nedges);
		return nedges;
  }
};

struct solver_ctx {
  edgetrimmer trimmer;
  bool mutatenonce;
  cl_uint2 *edges;
  graph<word_t> cg;
  cl_uint2 soledges[PROOFSIZE];
  std::vector<u32> sols; // concatenation of all proof's indices

  solver_ctx(const trimparams tp, bool mutate_nonce) : trimmer(tp), cg(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT) {
    edges   = new cl_uint2[MAXEDGES];
    mutatenonce = mutate_nonce;
  }

  void setheadernonce(char * const headernonce, const u32 len, const u32 nonce) {
    if (mutatenonce)
      ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &trimmer.sipkeys);
    sols.clear();
  }
  ~solver_ctx() {
    delete[] edges;
  }

  int findcycles(cl_uint2 *edges, u32 nedges) {
    cg.reset();
    for (u32 i = 0; i < nedges; i++)
      cg.add_compress_edge(edges[i].x, edges[i].y);
    for (u32 s = 0 ;s < cg.nsols; s++) {
       print_log("find Solution");
      for (u32 j = 0; j < PROOFSIZE; j++) {
        soledges[j] = edges[cg.sols[s][j]];
         print_log(" (%x, %x)", soledges[j].x, soledges[j].y);
      }
       print_log("\n");
      sols.resize(sols.size() + PROOFSIZE);
				cl_int clResult;
				clResult = clEnqueueWriteBuffer (trimmer.commandQueue, trimmer.recoveredges, CL_TRUE, 0, sizeof (cl_uint2) * PROOFSIZE, soledges, 0, NULL, NULL);
				checkOpenclErrors (clResult);

				int initV = 0;
				clResult = clEnqueueFillBuffer (trimmer.commandQueue, trimmer.uvnodes, &initV, sizeof (int), 0, trimmer.indexesSize, 0, NULL, NULL);
				checkOpenclErrors (clResult);

				clFinish (trimmer.commandQueue);
				cl_kernel recovery_kernel = clCreateKernel (trimmer.program, "Cuckaroo_Recovery", &clResult);
				clResult |= clSetKernelArg (recovery_kernel, 0, sizeof (cl_mem), (void *) &trimmer.dipkeys);
				clResult |= clSetKernelArg (recovery_kernel, 1, sizeof (cl_mem), (void *) &trimmer.uvnodes);
				clResult |= clSetKernelArg (recovery_kernel, 2, sizeof (cl_mem), (void *) &trimmer.recoveredges);
				checkOpenclErrors (clResult);

				cl_event event;
				size_t global_work_size[1], local_work_size[1];
				global_work_size[0] = trimmer.tp.recover.blocks * trimmer.tp.recover.tpb;
				local_work_size[0] = trimmer.tp.recover.tpb;
				clEnqueueNDRangeKernel (trimmer.commandQueue, recovery_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
				clFinish (trimmer.commandQueue);
				clResult = clEnqueueReadBuffer (trimmer.commandQueue, trimmer.uvnodes, CL_TRUE, 0, PROOFSIZE * sizeof (u32), &sols[sols.size () - PROOFSIZE], 0, NULL, NULL);
				checkOpenclErrors (clResult);
				qsort (&sols[sols.size () - PROOFSIZE], PROOFSIZE, sizeof (u32), cg.nonce_cmp);
    }
    return 0;
  }

  int solve() {
    u64 time0, time1;
    u32 timems,timems2;

    trimmer.abort = false;
    time0 = timestamp();
    u32 nedges = trimmer.trim();
    if (!nedges)
      return 0;
    if (nedges > MAXEDGES) {
      print_log("OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges-MAXEDGES, MAXEDGES);
      nedges = MAXEDGES;
    }
			cl_int clResult = clEnqueueReadBuffer (trimmer.commandQueue, trimmer.bufferB,
				CL_TRUE, 0, nedges * 8, edges, 0,
				NULL,
				NULL);
			checkOpenclErrors (clResult);
			//clResult = clEnqueueReadBuffer(trimmer.commandQueue, trimmer.dipkeys, CL_TRUE, 0, sizeof(siphash_keys), trimmer.dipkeys2, 0, NULL, NULL);
//    cudaMemcpy(edges, trimmer.bufferB, sizeof(uint2[nedges]), cudaMemcpyDeviceToHost);
    time1 = timestamp(); timems  = (time1 - time0) / 1000000;
    time0 = timestamp();
    findcycles(edges, nedges);
    time1 = timestamp(); timems2 = (time1 - time0) / 1000000;
    print_log("findcycles edges %d time %d ms total %d ms\n", nedges, timems2, timems+timems2);
    return sols.size() / PROOFSIZE;
  }

  void abort() {
    trimmer.abort = true;
  }
};

#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

typedef solver_ctx SolverCtx;

CALL_CONVENTION int run_solver(SolverCtx* ctx,
                               char* header,
                               int header_length,
                               u32 nonce,
                               u32 range,
                               SolverSolutions *solutions,
                               SolverStats *stats
                               )
{
  u64 time0, time1;
  u32 timems;
  u32 sumnsols = 0;
  int device_id;
  if (stats != NULL) {
//    cudaGetDevice(&device_id);
//    cudaDeviceProp props;
//    cudaGetDeviceProperties(&props, stats->device_id);
//    stats->device_id = device_id;
//    stats->edge_bits = EDGEBITS;
//    strncpy(stats->device_name, props.name, MAX_NAME_LEN);
  }

  if (ctx == NULL || !ctx->trimmer.initsuccess){
    print_log("Error initialising trimmer. Aborting.\n");
    print_log("Reason: %s\n", LAST_ERROR_REASON);
    if (stats != NULL) {
       stats->has_errored = true;
       strncpy(stats->error_reason, LAST_ERROR_REASON, MAX_NAME_LEN);
    }
    return 0;
  }

  for (u32 r = 0; r < range; r++) {
    time0 = timestamp();
    ctx->setheadernonce(header, header_length, nonce + r);
    print_log("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx->trimmer.sipkeys.k0, ctx->trimmer.sipkeys.k1, ctx->trimmer.sipkeys.k2, ctx->trimmer.sipkeys.k3);
    u32 nsols = ctx->solve();
    time1 = timestamp();
    timems = (time1 - time0) / 1000000;
    print_log("Time: %d ms, sols = %d\n", timems, nsols);
    for (unsigned s = 0; s < nsols; s++) {
      print_log("Solution");
      u32* prf = &ctx->sols[s * PROOFSIZE];
      for (u32 i = 0; i < PROOFSIZE; i++)
        print_log(" %jx", (uintmax_t)prf[i]);
      print_log("\n");
      if (solutions != NULL){
        solutions->edge_bits = EDGEBITS;
        solutions->num_sols++;
        solutions->sols[sumnsols+s].nonce = nonce + r;
        for (u32 i = 0; i < PROOFSIZE; i++) 
          solutions->sols[sumnsols+s].proof[i] = (u64) prf[i];
      }
      int pow_rc = verify(prf, ctx->trimmer.sipkeys);
      if (pow_rc == POW_OK) {
        print_log("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
        for (int i=0; i<32; i++)
          print_log("%02x", cyclehash[i]);
        print_log("\n");
      } else {
        print_log("FAILED due to %s\n", errstr[pow_rc]);
      }
    }
    sumnsols += nsols;
    if (stats != NULL) {
      stats->last_start_time = time0;
      stats->last_end_time = time1;
      stats->last_solution_time = time1 - time0;
    }
  }
  print_log("%d total solutions\n", sumnsols);
  return sumnsols > 0;
}

CALL_CONVENTION SolverCtx* create_solver_ctx(SolverParams* params) {
  trimparams tp;
  tp.ntrims = params->ntrims;
  tp.genA.blocks = params->genablocks;
  tp.genA.tpb = params->genatpb;
  tp.genB.tpb = params->genbtpb;
  tp.trim.tpb = params->trimtpb;
  tp.tail.tpb = params->tailtpb;
  tp.recover.blocks = params->recoverblocks;
  tp.recover.tpb = params->recovertpb;

  SolverCtx* ctx = new SolverCtx(tp, params->mutate_nonce);
	
  return ctx;
}

CALL_CONVENTION void destroy_solver_ctx(SolverCtx* ctx) {
  delete ctx;
}

CALL_CONVENTION void stop_solver(SolverCtx* ctx) {
  ctx->abort();
}

CALL_CONVENTION void fill_default_params(SolverParams* params) {
  trimparams tp;
  params->device = 0;
  params->ntrims = tp.ntrims;
  params->genablocks = std::min((u32)tp.genA.blocks, NEDGES/EDGE_BLOCK_SIZE/tp.genA.tpb);
  params->genatpb = tp.genA.tpb;
  params->genbtpb = tp.genB.tpb;
  params->trimtpb = tp.trim.tpb;
  params->tailtpb = tp.tail.tpb;
  params->recoverblocks = std::min((u32)tp.recover.blocks, NEDGES/EDGE_BLOCK_SIZE/tp.recover.tpb);
  params->recovertpb = tp.recover.tpb;
  params->cpuload = false;
}

int main(int argc, char **argv) {
  trimparams tp;
  u32 nonce = 0;
  u32 range = 1;
  u32 device = 0;
  char header[HEADERLEN];
  u32 len;
  int c;

  // set defaults
  SolverParams params;
  fill_default_params(&params);

  memset(header, 0, sizeof(header));
  while ((c = getopt(argc, argv, "scb:d:h:k:m:n:r:U:u:v:w:y:Z:z:")) != -1) {
    switch (c) {
      case 's':
        print_log("SYNOPSIS\n  cuda%d [-s] [-c] [-d device] [-h hexheader] [-m trims] [-n nonce] [-r range] [-U seedAblocks] [-u seedAthreads] [-v seedBthreads] [-w Trimthreads] [-y Tailthreads] [-Z recoverblocks] [-z recoverthreads]\n", NODEBITS);
        print_log("DEFAULTS\n  cuda%d -d %d -h \"\" -m %d -n %d -r %d -U %d -u %d -v %d -w %d -y %d -Z %d -z %d\n", NODEBITS, device, tp.ntrims, nonce, range, tp.genA.blocks, tp.genA.tpb, tp.genB.tpb, tp.trim.tpb, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
        exit(0);
      case 'c':
        params.cpuload = false;
        break;
      case 'd':
        device = params.device = atoi(optarg);
        break;
      case 'h':
        len = strlen(optarg)/2;
        assert(len <= sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i); // hh specifies storage of a single byte
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'm':
        params.ntrims = atoi(optarg) & -2; // make even as required by solve()
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'U':
        params.genablocks = atoi(optarg);
        break;
      case 'u':
        params.genatpb = atoi(optarg);
        break;
      case 'v':
        params.genbtpb = atoi(optarg);
        break;
      case 'w':
        params.trimtpb = atoi(optarg);
        break;
      case 'y':
        params.tailtpb = atoi(optarg);
        break;
      case 'Z':
        params.recoverblocks = atoi(optarg);
        break;
      case 'z':
        params.recovertpb = atoi(optarg);
        break;
    }
  }

  //checkCudaErrors(cudaGetDeviceCount(&nDevices));
 // assert(device < nDevices);
 // cudaDeviceProp prop;
 // checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  //u64 dbytes = prop.totalGlobalMem;
 // int dunit;
 // for (dunit=0; dbytes >= 102040; dbytes>>=10,dunit++) ;
 // print_log("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, prop.memoryClockRate/1000);
 // // cudaSetDevice(device);

  print_log("Looking for %d-cycle on cuckaroo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
  if (range > 1)
    print_log("-%d", nonce+range-1);
  print_log(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, params.ntrims, NX);

  SolverCtx* ctx = create_solver_ctx(&params);

  u64 bytes = ctx->trimmer.globalbytes();
  int unit;
  for (unit=0; bytes >= 102400; bytes>>=10,unit++) ;
  print_log("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);

  run_solver(ctx, header, sizeof(header), nonce, range, NULL, NULL);

  return 0;
}
