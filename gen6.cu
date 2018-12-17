//Editor: Michael Lukiman
//Spiking neuron network region implementation in CUDA with additional spatial winner-take-all dynamics
//GPU Architecture and Programming - Fall 2018


// As this is a faithful representation of neuronal spiking in a 'region' of the brain, we will do out best to cut down library use and use transparent operations:
#include <stdio.h> //Standard input-output
#include <stdlib.h> //StandardLibraryfunctions
#include <iostream> //For streaming input-output operations
#include <math.h> //math


__global__ excitatoryAsRegularSpiking(float * dev_a, float * dev_d) {
  int tid = threadIdx.x + ( blockIdx.x + blockDim.x);
  if (tid < excitatory) 
    {
    dev_a[tid]=0.02;
    dev_d[tid]=8.0;
    }
}

__global__ inhibitoryAsFastSpiking(float * dev_a, float * dev_d) {
  int tid = threadIdx.x + ( blockIdx.x + blockDim.x);
  if (tid < total - excitatory) 
    {
    int ind = tid + excitatory;
    dev_a[ind]=0.1;
    dev_d[ind]=2.0;
    }
}

__global__ excitatorySynapticWeights(float * dev_weights, int synapses) {
  tid =  threadIdx.x + ( blockIdx.x + blockDim.x);
  if (tid < excitatory)
    for(int j=0;j<synapses;j++)
      dev_weights[tid][j]=6.0;
}

__global__ inhibitorySynapticWeights(float * dev_weights, int synapses) {
  int tid = threadIdx.x + ( blockIdx.x + blockDim.x);
  if (tid < total - excitatory) 
    {
    int ind = tid + excitatory;
    for(int j=0;j<synapses;j++)
      weights[i][j]=-5.0;
}

__global__ synapticDerivatives(float * devs_w_derivs, int synapses) {
  int tid = threadIdx.x + ( blockIdx.x + blockDim.x);
  if (tid < total)
    for(int j=0;j<synapses;j++)
      w_derivs[i][j]=0.0;
}

__global__ initLT(float * dev_LTP, float * dev_LTD, int delay) {
  int tid = threadIdx.x + ( blockIdx.x + blockDim.x);
  if (tid < total)
    for(j=0;j<1+delay;j++)
      dev_LTP[tid][j]=0.0;
      dev_LTD[tid]=0.0;
}


__global__ initvu(float * dev_v, float * dev_u) {
  int tid = threadIdx.x + ( blockIdx.x + blockDim.x);
  if (tid < total)
    dev_v[tid]=-65.0;//Initialize v (resting membrane potential)
    dev_u[tid]=0.2*dev_v[tid];//initial values for u


__global__ warmup() {
  printf("Thank you.")
}

cudaStream_t stream1, stream2, stream3, stream4;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);
cudaStreamCreate(&stream4);

//Define parameters of the network:
const int excitatory=800;//Excitatory neurons(N_e)
const int inhibitory=200;//Inhibitory neurons(N_i)
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
float *dev_weights, *devs_w_derivs;   
short *dev_del_length;
short *dev_del_set;
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
uint i,j,k;
uint ii;
uint jj;
uint self;
uint rselect;

warmup<<<1,1>>>
cudaMalloc(&dev_a, sizeof(float)*total);
cudaMemcpy(dev_a, a, sizeof(float)*total, cudaMemcpyHostToDevice);
cudaMalloc(&dev_d, sizeof(float)*total);
cudaMemcpy(dev_d, d, sizeof(float)*total, cudaMemcpyHostToDevice);
excitatoryAsRegularSpiking<<<excBlocks128, threadsPerBlock,1>>>(dev_a,dev_d)//CUDA Stream 1 - Set excitatory as regular-spiking neurons
inhibitoryAsFastSpiking<<<inhBlocks128, threadsPerBlock,2>>>(dev_a, dev_d)//CUDA Stream 2 - Set inhibitory as fast-spiking neurons
cudaMemcpy(a, dev_a, sizeof(float)*total, cudaMemcpyDeviceToHost);
cudaMemcpy(d, dev_d, sizeof(float)*total, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

//Self-sort synapses:
for(i=0;i<total;i++)//Every neuron. -- CUDA note: would parallelize this, however it utilises a Host random function I edited (pstochastic()) - there are rand libraries for CUDA, but the overhead for cudarand is high. 
for(j=0;j<synapses;j++)//Every connected synapses.
  {
  do{
  self=0;

  if(i<excitatory)r=pstochastic(total);//Pick a random neuron
  else r=pstochastic(excitatory);//Pick a random excitatory neurons

  if(r==i)self=1;//Self selection

  //Not a recurrent network, so also prevent synapses connecting to self
  for(k=0;k<j;k++)
  if(ps_set[i][k]==rselect)//If the synapse connects to self
  self=1;

  }
  while(self==1);

  ps_set[i][j]=rselect;//This synapse included in the synapse set
}

cudaMalloc(&dev_weights, sizeof(float)*total*synapses);
cudaMalloc(&dev_w_derivs, sizeof(float)*total*synapses);
cudaMemcpy(dev_weights, weights, sizeof(float)*total*synapses, cudaMemcpyHostToDevice);
cudaDeviceSynchronize(); // Make sure those of non-zero stream do not launch before this is done. 
cudaMemcpyAsync(dev_w_derivs, w_derivs, sizeof(float)*total*synapses, cudaMemcpyHostToDevice, stream3); // This will potentially overlap the derivatives transfer with the two following kernels, and is the same stream as the third kernel that eventually needs it. 
excitatorySynapticWeights<<<excBlocks128,threadsPerBlock,stream1>>>(dev_weights, synapses);//CUDA Stream 1 - Initialize excitatory synaptic weights
inhibitorySynapticWeights<<<inhBlocks128,threadsPerBlock,stream2>>>(dev_weights, synapses);//CUDA Stream 2 - Initialize inhibitory weights
synapticDerivatives<<<totBlocks128, threadsPerBlock,stream3>>>(dev_w_derivs, synapses)//CUDA Stream 3 - Initialize synaptic derivatives
// Using streams to hide memory transfer latency to some degree. 
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
cudaMemcpyAsync(weights, dev_weights, sizeof(float)*total*synapses, cudaMemcpyDeviceToHost, stream1); 
cudaMemcpyAsync(w_derivs, dev_w_derivs, sizeof(float)*total*synapses, cudaMemcpyDeviceToHost, stream3); // Since this is in stream 3, it will implicitly wait for the derivative operation (also in stream 3) to complete. 
cudaDeviceSynchronize(); // Explicit synchronization for non-zero streams. Host code to follow:

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
    if(ps_set[j][k]==i)//This is a presynaptic neuron
      {
      pre_input[i][pre_neuron[i]]=j;//Register it to the set

      for(jj=0;jj<delay;jj++)//Every delay
      for(ii=0;ii<del_length[j][jj];ii++)
      if(ps_set[j][ del_set[j][jj][ii] ]==i)
        pre_delay[i][pre_neuron[i]]=jj;
      pre_weights[i][pre_neuron[i]]=&weights[j][k];//Presynaptic weight assigned to relevant synaptic weights
      pre_w_derivs[i][pre_neuron[i]++]=&w_derivs[j][k];//Likewise with derivatives
      } // I had originally attempted to parallelize this, but the transfer of all matrices proves to be too costly for their latency and memory access pattern.
  }

//Initialize longterm values and membrane potentials
initLT<<<totBlocks128, threadsPerBlock, stream1>>>(dev_LTP, delay);//CUDA note: could very well be put in same kernel as operations below, but want to take advantage of parallel streams. 
initvu<<<totBlocks128, threadsPerBlock, stream2>>>(dev_v, dev_u);//CUDA
cudaDeviceSynchronize(); // Explicit non-default stream synchronize.

num_fired=1;//spike timings
spike[0][0]=-delay;//dummy spike with negative delay interval for warmup
spike[0][1]=0;//dummy spike

}

int main()
{

uint i,j,k;//Loop counters
uint sec,t;//seconds, milliseconds
float	inputs[total];//Inputs to neurons!
FILE	*fs;//File pointer

initialize();

for(sec=0;sec<60;sec++) {//plot for 1minute(60s)
for(t=0;t<1000;t++)//plot for 1sec(1000ms)
  {
  for(i=0;i<total;i++)inputs[i]=0.0;//Fresh input
  for(k=0;k<total/1000;k++) inputs[pstochastic(total)]=20.0;//Noisy input

  for(i=0;i<total;i++)
    if(v[i]>=30)//Passing the threshold:
      {
      v[i]=-65.0;//Zero the voltage
      u[i]+=d[i];//Refractory period
      LTpot[i][t+delay]=0.1;// Update potentiation
      LTdep[i]=0.12;//Update depression

      for(j=0;j<pre_neuron[i];j++)
        *pre_w_derivs[i][j]+=LTpot[pre_input[i][j]][t+delay-pre_delay[i][j]-1];//Spike after presynaptic spike
      spike[num_fired][0]=t;
      spike[num_fired++][1]=i;
      if(num_fired==hz){ std::cout<<"Too many spikes at t="<<t<<"(ignoring all)"; num_fired=1; }
      }

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

  for(i=0;i<total;i++)
    {
    v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+inputs[i]);//Izkevich formulae
    v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+inputs[i]);//Repeat
    u[i]+=a[i]*(0.2*v[i]-u[i]);
    LTpot[i][t+delay+1]=0.95*LTpot[i][t+delay];
    LTdep[i]*=0.95;
    }
  }

std::cout<<"sec="<<sec<<",firingrate="<<float(num_fired)/total<<"\n";
fs=fopen("spikes.dat","w");
for(i=1;i<num_fired;i++)
if(spike[i][0]>=0)
  fprintf(fs,"%d%d\n",spike[i][0],spike[i][1]);
fclose(fs);

//Next second
for(i=0;i<total;i++)
for(j=0;j<delay+1;j++)
  LTpot[i][j]=LTpot[i][1000+j];

k=num_fired-1;
while(1000-spike[k][0]<delay)k--;

for(i=1;i<num_fired-k;i++)
  {
  spike[i][0]=spike[k+i][0]-1000;
  spike[i][1]=spike[k+i][1];
  }
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
return 1;
}
