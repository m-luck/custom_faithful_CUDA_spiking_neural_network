//Editor: Michael Lukiman
//Spiking neuron network region implementation in CUDA with additional spatial winner-take-all dynamics
//GPU Architecture and Programming - Fall 2018


// As this is a faithful representation of neuronal spiking in a 'region' of the brain, we will do out best to cut down library use and use transparent operations:
#include <stdio.h> //Standard input-output
#include <stdlib.h> //StandardLibraryfunctions
#include <iostream> //For streaming input-output operations
#include <cuda.h> // Ensure enabled parallelization
#include <math.h> //math



__global__ void excitatoryAsRegularSpiking(float * dev_a, float * dev_d, int total, int excitatory) {
  int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
  if (tid < excitatory)
    {
    dev_a[tid]=0.02;
    dev_d[tid]=8.0;
    }
}

__global__ void inhibitoryAsFastSpiking(float * dev_a, float * dev_d, int total, int excitatory) {
  int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
  if (tid < total - excitatory)
    {
    int ind = tid + excitatory;
    dev_a[ind]=0.1;
    dev_d[ind]=2.0;
    }
}

__global__ void excitatorySynapticWeights(float * dev_weights, int synapses, int total, int excitatory) {
  int tid =  threadIdx.x + ( blockIdx.x * blockDim.x);
  if (tid < excitatory)
    for(int j=0;j<synapses;j++)
      dev_weights[tid*synapses+j]=6.0;
}

__global__ void inhibitorySynapticWeights(float * dev_weights, int synapses, int total, int excitatory) {
  int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
  if (tid < total - excitatory)
    {
    int ind = tid + excitatory;
    for(int j=0;j<synapses;j++)
      dev_weights[ind*synapses+j]=-5.0;
    }
}

__global__ void synapticDerivatives(float * dev_w_derivs, int synapses, int total, int excitatory) {
  int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
  if (tid < total)
    for(int j=0;j<synapses;j++)
      dev_w_derivs[tid*synapses+j]=0.0;
}

__global__ void initLT(float * dev_LTP, float * dev_LTD, int delay, int total, int excitatory) {
  int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
  if (tid < total)
    for(int j=0;j<1+delay;j++){
      dev_LTP[tid*(1001+delay)+j]=0.0;
      dev_LTD[tid]=0.0;}
}


__global__ void initvu(float * dev_v, float * dev_u, int total, int excitatory) {
  int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
  if (tid < total)
    dev_v[tid]=-65.0;//Initialize v (resting membrane potential)
    dev_u[tid]=0.2*dev_v[tid];//initial values for u
  }

__global__ void updateVoltages(int i, int t, int total, int delay, float * v, float * u, float * inputs, float * a, float * LTpot, float * LTdep)
    {
    int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
    if (tid<total) {
      v[tid]+=0.5*((0.04*v[tid]+5)*v[tid]+140-u[tid]+inputs[tid]);//Izkevtidch formulae
      v[tid]+=0.5*((0.04*v[tid]+5)*v[tid]+140-u[tid]+inputs[tid]);//Repeat
      u[tid]+=a[tid]*(0.2*v[tid]-u[tid]);
      LTpot[tid*(1001+delay)+t+delay+1]=0.95*LTpot[tid*(1001+delay)+t+delay];
      LTdep[tid]*=0.95;
  }
}

__global__ void updatePotentiation(int total, int delay, float * LTpot)    {
    int tid = threadIdx.x + ( blockIdx.x * blockDim.x);
    if (tid<total)
      for(int j=0;j<delay+1;j++)
        LTpot[tid*(1001+delay)+j]=LTpot[tid*(1001+delay)+1000+j];
}

__global__ void warmup() {
  printf("Thank you.");
}

//Define parameters of the network:
const int excitatory=750;//Excitatory neurons(N_e)
const int inhibitory=250;//Inhibitory neurons(N_i)
const int total=excitatory+inhibitory;//Total Exc.+Inh. neurons
const int synapses=100;//synapses per neuron
const int delay=20;//in milliseconds, top axonal conduction delay
const int hz=total*100;//Upper bound of firingrate
const float max_weight=10.0;//Top synaptic weight

//Neuronal dynamics
float a[total];
float d[total];

//Activity variables
float v[total];
float u[total];

int num_fired;//amount of fired neurons
int spike[hz][2];//Timing of spikes, max of hz limit

//Spike-timing dependent variables (LTP,LTD)
float LTpot[total][delay+1001];//Longterm potentiation
float LTdep[total];//Longterm depression

//Turn these relationships into data arrays
int ps_set[total][synapses];//Matrix holding the post-synaptic neurons(synapses) of each neuron(total)
float weights[total][synapses];//Matrix holding the weights of each synapse
float w_derivs[total][synapses];//Matrix holding the derivative of each above weight
float axonal_dist[total][synapses];//Matrix holding actuation location distance from synapse
short del_length[total][delay];//Matrix holding the delay values of each neuron
short del_set[total][delay][synapses];//Matrix holding the del_set to each synapse from each neuron
int pre_neuron[total];//Index of presynaptic information
int pre_input[total][synapses*3];//Presynaptic inputs
int pre_delay[total][synapses*3];//Presynaptic del_set
float *pre_weights[total][synapses*3];//Presynaptic weights
float *pre_w_derivs[total][synapses*3];//Presynaptic derivatives

//CUDA Business:
const int threadsPerBlock = 128;
const int excBlocks128 = (excitatory + 128 - 1)/128;
const int inhBlocks128 = (inhibitory + 128 - 1)/128;
const int totBlocks128 = (total + 128 -1)/128;

//These are all analogs of the Host items above:
int *dev_ps_set;
float *dev_weights, *dev_w_derivs;
short *dev_del_length;
short *dev_del_set;
float *dev_axonal_dist;
int *dev_pre_neuron, *dev_pre_input, *dev_pre_delay;
float *dev_pre_weights, *dev_pre_w_derivs;
float *dev_LTP, *dev_LTD;
float *dev_a, *dev_d;
float *dev_v, *dev_u;
float *dev_inputs;
int *dev_num_fired;
int *dev_spike;

int pstochastic(int n) { // Pseudo-stochastic/random
  return rand() % (int)(n);
}

void initialize()
{

cudaStream_t stream1, stream2, stream3, stream4;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);
cudaStreamCreate(&stream4);

int i,j,k;
int ii;
int jj;
int self;
int rselect;

warmup<<<1,1>>>();
cudaMalloc(&dev_a, sizeof(float)*total);
// cudaMemcpy(dev_a, a, sizeof(float)*total, cudaMemcpyHostToDevice);
cudaMalloc(&dev_d, sizeof(float)*total);
// cudaMemcpy(dev_d, d, sizeof(float)*total, cudaMemcpyHostToDevice);
cudaDeviceSynchronize();
excitatoryAsRegularSpiking<<<excBlocks128, threadsPerBlock,0,stream1>>>(dev_a,dev_d,total,excitatory);//CUDA Stream 1 - Set excitatory as regular-spiking neurons
inhibitoryAsFastSpiking<<<inhBlocks128, threadsPerBlock,0,stream2>>>(dev_a, dev_d,total,excitatory);//CUDA Stream 2 - Set inhibitory as fast-spiking neurons
cudaDeviceSynchronize();
cudaMemcpy(a, dev_a, sizeof(float)*total, cudaMemcpyDeviceToHost);
cudaMemcpy(d, dev_d, sizeof(float)*total, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

//
// for(i=0;i<excitatory;i++)a[i]=0.02;//Set excitatory as regular-spiking neurons
//
// for(i=excitatory;i<total;i++)a[i]=0.1;//Set inhibitory as fast-spiking neurons
//
// for(i=0;i<excitatory;i++)d[i]=8.0;//Set excitatory as regular-spiking neurons
//
// for(i=excitatory;i<total;i++)d[i]=2.0;//Set inhibitory as fast-spiking neurons


//Self-sort synapses:
for(i=0;i<total;i++)//Every neuron. -- CUDA note: would parallelize this, however it utilises a Host random function I edited (pstochastic()) - there are rand libraries for CUDA, but the overhead for cudarand is high.
for(j=0;j<synapses;j++)//Every connected synapses.
  {
  do{
  self=0;

  if(i<excitatory)rselect=pstochastic(total);//Pick a random neuron
  else rselect=pstochastic(excitatory);//Pick a random excitatory neurons

  if(rselect==i)self=1;//Self selection

  //Not a recurrent network, so also prevent synapses connecting to self
  for(k=0;k<j;k++)
  if(ps_set[i][k]==rselect)//If the synapse connects to self
  self=1;

  }
  while(self==1);

  ps_set[i][j]=rselect;//This synapse included in the synapse set

  // Here is where the spatial dynamics take place -
  axonal_dist[i][j]=(float)log2f(pstochastic(500000));//Up to 50 centimeters away (diameter of the human brain), or 500,000 microns! Then adjusted for nonlinear loss due to travel resistance.
}

cudaMalloc(&dev_weights, sizeof(float)*total*synapses);
cudaMalloc(&dev_w_derivs, sizeof(float)*total*synapses);
// cudaMemcpy(dev_weights, weights, sizeof(float)*total*synapses, cudaMemcpyHostToDevice);
cudaDeviceSynchronize(); // Make sure those of non-zero stream do not launch before this is done.
// cudaMemcpyAsync(dev_w_derivs, w_derivs, sizeof(float)*total*synapses, cudaMemcpyHostToDevice, stream3); // This will potentially overlap the derivatives transfer with the two following kernels, and is the same stream as the third kernel that eventually needs it.
excitatorySynapticWeights<<<excBlocks128,threadsPerBlock,0,stream1>>>(dev_weights, synapses,total,excitatory);//CUDA Stream 1 - Initialize excitatory synaptic weights
inhibitorySynapticWeights<<<inhBlocks128,threadsPerBlock,0,stream2>>>(dev_weights, synapses,total,excitatory);//CUDA Stream 2 - Initialize inhibitory weights
synapticDerivatives<<<totBlocks128, threadsPerBlock,0,stream3>>>(dev_w_derivs, synapses,total,excitatory);//CUDA Stream 3 - Initialize synaptic derivatives
// Using streams to hide memory transfer latency to some degree.
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
cudaMemcpy(weights, dev_weights, sizeof(float)*total*synapses, cudaMemcpyDeviceToHost);
cudaMemcpy(w_derivs, dev_w_derivs, sizeof(float)*total*synapses, cudaMemcpyDeviceToHost); // Since this is in stream 3, it will implicitly wait for the derivative operation (also in stream 3) to complete.
cudaDeviceSynchronize(); // Explicit synchronization for non-zero streams. Host code to follow:

//
// //Initialize excitatory synaptic weights
// for(i=0;i<excitatory;i++)
// for(j=0;j<synapses;j++)
// weights[i][j]=6.0;
//
// //Initialize inhibitory weights
// for(i=excitatory;i<total;i++)
// for(j=0;j<synapses;j++)
// weights[i][j]=-5.0;
//
// //Initialize synaptic derivatives
// for(i=0;i<total;i++)
// for(j=0;j<synapses;j++)
// w_derivs[i][j]=0.0;

for(i=0;i<total;i++)//For every neuron
  {
  short ind=0;//Keep track of its index

  if(i<excitatory)//If the neuron is excitatory
    {
    //Update delay lengths
    for(j=0;j<delay;j++)
    {
    del_length[i][j]=synapses/delay;//Allocate equal intervals
      //Update del_set via delay lengths
      for(k=0;k<del_length[i][j];k++)
      del_set[i][j][k]=ind++;//CUDA note: since this manually adjusts an index, we are wary to parallelize loops around between this and ind=0.
    }
    }

  else
    {
    //Set all del_set to 1ms from delay start
    for(j=0;j<delay;j++)
    del_length[i][j]=0;

    del_length[i][0]=synapses;

    for(k=0;k<del_length[i][0];k++)
    del_set[i][0][k]=ind++;//CUDA note: here as well.

    }
}

for(i=0;i<total;i++)
  {
  pre_neuron[i]=0;
  for(j=0;j<excitatory;j++)
  for(k=0;k<synapses;k++)
    if(ps_set[j][k]==i)
      {
      pre_input[i][pre_neuron[i]]=j;//Register it to the set

      for(jj=0;jj<delay;jj++)//Every delay
      for(ii=0;ii<del_length[j][jj];ii++)
      if(ps_set[j][ del_set[j][jj][ii] ]==i)
        pre_delay[i][pre_neuron[i]]=jj;
      pre_weights[i][pre_neuron[i]]=&weights[j][k];//Presynaptic weight assigned to relevant synaptic weights
      pre_w_derivs[i][pre_neuron[i]++]=&w_derivs[j][k];//Likewise with derivatives
    }
  }

//Initialize longterm values and membrane potentials
cudaMalloc(&dev_LTP, sizeof(float)*total*(1001+delay));
cudaMalloc(&dev_LTD, sizeof(float)*total);
cudaMalloc(&dev_v, sizeof(float)*total);
cudaMalloc(&dev_u, sizeof(float)*total);
cudaDeviceSynchronize();
initLT<<<totBlocks128, threadsPerBlock,0, stream1>>>(dev_LTP, dev_LTD, delay,total,excitatory);//CUDA note: could very well be put in same kernel as operations below, but want to take advantage of parallel streams.
initvu<<<totBlocks128, threadsPerBlock,0, stream2>>>(dev_v, dev_u,total,excitatory);//CUDA
cudaDeviceSynchronize(); // Explicit non-default stream synchronize.
cudaMemcpy(LTpot, dev_LTP, sizeof(float)*total*(1001+delay), cudaMemcpyDeviceToHost);
cudaMemcpy(LTdep, dev_LTD, sizeof(float)*total, cudaMemcpyDeviceToHost);
cudaMemcpy(v, dev_v, sizeof(float)*total, cudaMemcpyDeviceToHost);
cudaMemcpy(u, dev_u, sizeof(float)*total, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

// //Initialize longterm potentiation values
// for(i=0;i<total;i++)
// for(j=0;j<1+delay;j++)
// LTpot[i][j]=0.0;
//
// for(i=0;i<total;i++)	LTdep[i]=0.0;//Initialize longterm depression values
//
// for(i=0;i<total;i++)	v[i]=-65.0;//Initialize v (resting membrane potential)
//
// for(i=0;i<total;i++)	u[i]=0.2*v[i];//initial values for u


num_fired=1;//spike timings
spike[0][0]=-delay;//dummy spike with negative delay interval for warmup
spike[0][1]=0;//dummy spike

}

int main()
{

cudaMalloc(&dev_inputs, sizeof(float)*total);

int i,j,k;//Loop counters
int sec,t;//seconds, milliseconds
float	inputs[total];
FILE	*fs;//File pointer to trace
FILE	*fs2D;//File pointer to individual neuron tracing!

initialize();
fs=fopen("trace.out","w");
fs2D=fopen("slice_2D_trace.out","w");

for(sec=0;sec<3;sec++) {//plot for 3s)
for(t=0;t<1000;t++)//plot for 1sec(1000ms)
  {
  for(i=0;i<total;i++)inputs[i]=0.0;//Fresh input
  for(k=0;k<total/1000;k++) inputs[pstochastic(total)]=20.0;//Noisy input

  for(i=0;i<total;i++)
    if(v[i]>=30)//Passing the voltage threshold:
      {
      v[i]=-65.0;//"Zero" the voltage to resting potential
      u[i]+=d[i];//Refractory period
      LTpot[i][t+delay]=0.1;// Update potentiation
      LTdep[i]=0.12;//Update depression

      for(j=0;j<pre_neuron[i];j++)
        *pre_w_derivs[i][j]+=(1/axonal_dist[i][j])* // Adjusted arbitrary voltage loss through travel. CUDA note: The pointer in this logic makes it circuitous to parallelize.
                              LTpot[pre_input[i][j]][t+
                              delay-pre_delay[i][j]-1];//Spike after presynaptic spike

      fprintf(fs2D,"%d:%d,\n",t,i);
      spike[num_fired][0]=t;
      spike[num_fired][1]=i;
      num_fired++;
      if(num_fired==hz){ printf("Overloaded spikes at t=%d", t); num_fired=1; }
      }

      fprintf(fs,"%d\n",num_fired);

  k=num_fired;
  while(t-spike[--k][0]<delay)
    {
    for(j=0;j<del_length[spike[k][1]][t-spike[k][0]];j++)
      {
      i=ps_set[spike[k][1]][del_set[spike[k][1]][t-spike[k][0]][j]];
      inputs[i]+= weights[ spike[k][1] ] [ del_set[ spike[k][1] ] [ t-spike[k][0] ] [j] ];
      if(spike[k][1]<excitatory)//Spike before post-synaptic
      	w_derivs[spike[k][1]][del_set[spike[k][1]][t-spike[k][0]][j]]-=LTdep[i];
      }
    }

  //Update voltages per Iz formula

  cudaMemcpy(dev_a, a, sizeof(float)*total, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_u, u, sizeof(float)*total, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_v, v, sizeof(float)*total, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_LTD, LTdep, sizeof(float)*total, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_LTP, LTpot, sizeof(float)*total*(1001+delay), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_inputs, inputs, sizeof(float)*total, cudaMemcpyHostToDevice);
  updateVoltages<<<totBlocks128, threadsPerBlock>>>(i,t,total, delay, dev_v,dev_u,dev_inputs,dev_a,dev_LTP,dev_LTD);
  cudaMemcpy(inputs, dev_inputs, sizeof(float)*total, cudaMemcpyDeviceToHost);
  cudaMemcpy(LTpot, dev_LTP, sizeof(float)*total*(1001+delay), cudaMemcpyDeviceToHost);
  cudaMemcpy(LTdep, dev_LTD, sizeof(float)*total, cudaMemcpyDeviceToHost);
  cudaMemcpy(v, dev_v, sizeof(float)*total, cudaMemcpyDeviceToHost);
  cudaMemcpy(u, dev_u, sizeof(float)*total, cudaMemcpyDeviceToHost);
  cudaMemcpy(a, dev_a, sizeof(float)*total, cudaMemcpyDeviceToHost);
  //
  // for(i=0;i<total;i++)
  //   {
  //   v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+inputs[i]);//Izkevich formulae
  //   v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+inputs[i]);//Repeat
  //   u[i]+=a[i]*(0.2*v[i]-u[i]);
  //   LTpot[i][t+delay+1]=0.95*LTpot[i][t+delay];
  //   LTdep[i]*=0.95;
  //   }

  }

printf("Time at %d firing Rate = %.5f\n",sec,(float)(num_fired)/total);

cudaMemcpy(dev_LTP, LTpot, sizeof(float)*total*(1001+delay), cudaMemcpyHostToDevice);
updatePotentiation<<<totBlocks128,threadsPerBlock>>>(total, delay, dev_LTP);
cudaMemcpy(LTpot, dev_LTP, sizeof(float)*total*(1001+delay), cudaMemcpyDeviceToHost);

//Next second
// for(i=0;i<total;i++)
// for(j=0;j<delay+1;j++)
//   LTpot[i][j]=LTpot[i][1000+j];

k=num_fired-1;
while(1000-spike[k][0]<delay)k--;

for(i=1;i<num_fired-k;i++)
  {
  spike[i][0]=spike[k+i][0]-1000;
  spike[i][1]=spike[k+i][1];
  }
// YOU ARE HERE AT UPDATE SPIKE
// THEN UPDATE EXCITATORY CONNECTIONS
  
num_fired=num_fired-k;

for(i=0;i<excitatory;i++)	//Update excitatory connections
for(j=0;j<synapses;j++)
  {
  weights[i][j]+=0.01+w_derivs[i][j];
  w_derivs[i][j]*=0.9;
  if(weights[i][j]>max_weight) weights[i][j]=max_weight;
  if(weights[i][j]<0) weights[i][j]=0.0;
  }

}
fclose(fs2D);
fclose(fs);
return 1;
}
