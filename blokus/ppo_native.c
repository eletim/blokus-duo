/* Ordered all-placement scan and exact float32 candidate packing.
   No fast-math, RNG, state cache, or mutable global data. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static int pc(uint64_t x) { return __builtin_popcountll(x); }
int ppo_scan(const uint64_t *ctx, uint32_t remaining, int total,
 const uint64_t *geometry, const int32_t *pieces, const float *fixed,
 int32_t *ids, float *out, int pack) {
 int fc=0,own=0,other=0,n=0,rc=__builtin_popcount(remaining);
 for(int w=0;w<4;w++){own+=pc(ctx[w]);other+=pc(ctx[4+w]);fc+=pc(ctx[8+w]);}
 for(int i=0;i<total;i++) {
  if(!(remaining & (1u<<pieces[i])))continue;
  const uint64_t *bits=geometry+12*i,*edges=bits+4,*corners=bits+8;
  int touch=0,blocked=0;
  for(int w=0;w<4;w++){
   blocked|=(bits[w]&(ctx[w]|ctx[4+w]|ctx[16+w]))!=0;
   touch|=(bits[w]&ctx[8+w])!=0;
  }
  if(blocked||!touch)continue;
  ids[n]=i;
  if(pack){
   float *x=out+39*n;memcpy(x,fixed+39*i,39*sizeof(float));
   int nf=0,cover=0,edgefront=0,cornerempty=0,edgeother=0;
   for(int w=0;w<4;w++){
    uint64_t occ=ctx[w]|ctx[4+w];
    nf+=pc((ctx[8+w]|corners[w])&~(occ|bits[w]|ctx[16+w]|edges[w]));
    cover+=pc(bits[w]&ctx[12+w]);edgefront+=pc(edges[w]&ctx[12+w]);
    cornerempty+=pc(corners[w]&~occ&~bits[w]);edgeother+=pc(edges[w]&ctx[4+w]);
   }
   x[1]=(float)((nf-fc)/10.0);x[2]=(float)(cover/5.0);x[5]=(float)(edgefront/10.0);
   x[6]=(float)(nf/40.0);x[7]=(float)(cornerempty/20.0);x[9]=(float)(rc/21.0);
   x[10]=(float)((own-other)/89.0);x[11]=(float)(edgeother/10.0);
  }
  n++;
 }
 return n;
}
