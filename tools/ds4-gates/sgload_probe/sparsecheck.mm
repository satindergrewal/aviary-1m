// Does THIS device support placement-sparse BUFFERS? Grok (#2430) challenged my claim that
// reserve-VA-then-commit is impossible for buffers on Metal. I had grepped the OLD sparse
// path (MTLResourceStateCommandEncoder, texture-only) and generalised from its absence.
// Placement sparse is a separate, newer mechanism. This asks the hardware.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main() {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) { printf("no device\n"); return 2; }
        printf("device: %s\n", d.name.UTF8String);
        printf("recommendedMaxWorkingSetSize: %.1f GiB\n", d.recommendedMaxWorkingSetSize/1073741824.0);
        printf("hasUnifiedMemory: %s\n", d.hasUnifiedMemory ? "YES" : "NO");
        if (@available(macOS 26.4, *)) {
            printf("supportsPlacementSparse: %s\n", d.supportsPlacementSparse ? "*** YES ***" : "NO");
        } else {
            printf("supportsPlacementSparse: <API needs macOS 26.4; this SDK/OS is older>\n");
        }
        for (int f = (int)MTLGPUFamilyApple9; f >= (int)MTLGPUFamilyApple7; --f) {
            if ([d supportsFamily:(MTLGPUFamily)f]) { printf("highest GPU family: Apple%d\n", f - (int)MTLGPUFamilyApple1 + 1); break; }
        }
        return 0;
    }
}
