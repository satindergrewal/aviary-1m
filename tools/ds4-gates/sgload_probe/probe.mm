// Host side for the DS4 simdgroup_load pitch probe. Build: see build.sh
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <set>

int main(int argc, char ** argv) {
    // SRC_W = source row stride in ELEMENTS (the "true" pitch). LOAD/STORE pitches are argv.
    const uint32_t SRC_W       = argc > 1 ? (uint32_t) atoi(argv[1]) : 8;
    const uint32_t LOAD_PITCH  = argc > 2 ? (uint32_t) atoi(argv[2]) : 8;
    const uint32_t STORE_PITCH = argc > 3 ? (uint32_t) atoi(argv[3]) : 8;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }

        NSError * err = nil;
        // Compile at RUNTIME (as ggml-metal does) -- needs no offline Metal toolchain.
        NSString * src = [NSString stringWithContentsOfFile:@"probe.metal"
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "read probe.metal: %s\n", err.localizedDescription.UTF8String); return 2; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 2; }

        id<MTLFunction> fn = [lib newFunctionWithName:@"sgload_probe"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 2; }

        // Source: 16 rows of SRC_W so an over-long pitch reads real data, not garbage.
        const uint32_t SRC_ROWS = 16;
        const uint32_t src_n = SRC_ROWS * SRC_W;
        id<MTLBuffer> bsrc = [dev newBufferWithLength:src_n*sizeof(float) options:MTLResourceStorageModeShared];
        float * s = (float *) bsrc.contents;
        for (uint32_t r = 0; r < SRC_ROWS; ++r)
            for (uint32_t c = 0; c < SRC_W; ++c)
                s[r*SRC_W + c] = (float)(r*100 + c);   // encodes its own (row,col)

        const uint32_t dst_n = 8*STORE_PITCH + 64;
        id<MTLBuffer> bdst = [dev newBufferWithLength:dst_n*sizeof(float) options:MTLResourceStorageModeShared];
        float * d = (float *) bdst.contents;
        for (uint32_t i = 0; i < dst_n; ++i) d[i] = -999.0f;   // untouched marker

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bsrc offset:0 atIndex:0];
        [enc setBuffer:bdst offset:0 atIndex:1];
        [enc setBytes:&LOAD_PITCH  length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&STORE_PITCH length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "gpu: %s\n", cb.error.localizedDescription.UTF8String); return 2; }

        printf("SRC_W=%u LOAD_PITCH=%u STORE_PITCH=%u\n", SRC_W, LOAD_PITCH, STORE_PITCH);
        std::set<int> rows_seen;
        for (uint32_t r = 0; r < 8; ++r) {
            printf("  frag row %u: ", r);
            for (uint32_t c = 0; c < 8; ++c) {
                float v = d[r*STORE_PITCH + c];
                if (v == -999.0f) { printf("  ....."); continue; }
                printf(" %6.0f", v);
                if (v >= 0) rows_seen.insert(((int) v) / 100);
            }
            printf("\n");
        }
        printf("DISTINCT SOURCE ROWS REACHED: %zu  {", rows_seen.size());
        for (int r : rows_seen) printf(" %d", r);
        printf(" }\n");
        // The gate is "source rows 0..7 EXACTLY" -- not merely "eight distinct rows".
        // A doubled load pitch reaches eight distinct rows (0,2,..,14), half of them past
        // the intended tile, and would pass a naive count. It must FAIL.
        bool exact = (rows_seen.size() == 8);
        for (int r = 0; r < 8 && exact; ++r) exact = rows_seen.count(r) > 0;
        printf("VERDICT: %s\n", exact ? "ROWS-0..7-EXACT" : "WRONG-ROWS (tile misread)");
        return exact ? 0 : 1;
    }
}
