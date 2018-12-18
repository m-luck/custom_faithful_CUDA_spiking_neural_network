//Editor: Michael Lukiman
//Izhikevich spiking neuron network implementation in CUDA with added spatial winner-take-all dynamics
//GPU Architecture and Programming - Fall 2018

#include <stdio.h> //Standard input-output
#include <stdlib.h> //StandardLibraryfunctions
#include <iostream> //For streaming input-output operations
#include <math.h> //math

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

int pstochastic(int n) { // Pseudo-stochastic/random
  return rand() % (int)(n);
}

void initialize()
{
int i,j,k;
int jj;
int dd;
int self;
int r;

for(i=0;i<excitatory;i++)a[i]=0.02;//Set excitatory as regular-spiking neurons

for(i=excitatory;i<total;i++)a[i]=0.1;//Set inhibitory as fast-spiking neurons

for(i=0;i<excitatory;i++)d[i]=8.0;//Set excitatory as regular-spiking neurons

for(i=excitatory;i<total;i++)d[i]=2.0;//Set inhibitory as fast-spiking neurons

//Self-sort synapses:
for(i=0;i<total;i++)//Every neuron
for(j=0;j<synapses;j++)//Every weight of connected synapse
  {
  do{
  self=0;

  if(i<excitatory)r=pstochastic(total);//Pick a random neuron
  else r=pstochastic(excitatory);//Pick a random excitatory neurons

  if(r==i)self=1;//Self selection

  //Not a recurrent network, so also prevent synapses connecting to self
  for(k=0;k<j;k++)
  if(ps_set[i][k]==r)//If the synapse connects to self
  self=1;

  }
  while(self==1);

  ps_set[i][j]=r;//This synapse is randomly assigned
}

//Initialize excitatory synaptic weights
for(i=0;i<excitatory;i++)
for(j=0;j<synapses;j++)
weights[i][j]=6.0;

//Initialize inhibitory weights
for(i=excitatory;i<total;i++)
for(j=0;j<synapses;j++)
weights[i][j]=-5.0;

//Initialize synaptic derivatives
for(i=0;i<total;i++)
for(j=0;j<synapses;j++)
w_derivs[i][j]=0.0;

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
      del_set[i][j][k]=ind++;
    }
    }

  else
    {
    //Set all del_set to 1ms from delay start
    for(j=0;j<delay;j++)
    del_length[i][j]=0;

    del_length[i][0]=synapses;

    for(k=0;k<del_length[i][0];k++)
    del_set[i][0][k]=ind++;

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

      for(dd=0;dd<delay;dd++)//Every delay
      for(jj=0;jj<del_length[j][dd];jj++)
      if(ps_set[j][ del_set[j][dd][jj] ]==i)
        pre_delay[i][pre_neuron[i]]=dd;
      pre_weights[i][pre_neuron[i]]=&weights[j][k];//Presynaptic weight assigned to relevant synaptic weights
      pre_w_derivs[i][pre_neuron[i]++]=&w_derivs[j][k];//Likewise with derivatives
      }
  }

//Initialize longterm potentiation values
for(i=0;i<total;i++)
for(j=0;j<1+delay;j++)
LTpot[i][j]=0.0;

for(i=0;i<total;i++)	LTdep[i]=0.0;//Initialize longterm depression values

for(i=0;i<total;i++)	v[i]=-65.0;//Initialize v (resting membrane potential)

for(i=0;i<total;i++)	u[i]=0.2*v[i];//initial values for u

num_fired=1;//spike timings
spike[0][0]=-delay;//dummy spike with negative delay interval for warmup
spike[0][1]=0;//dummy spike

}

int main()
{

int i,j,k;//Loop counters
int sec,t;//seconds, milliseconds
float	inputs[total];//Inputs to neurons!
FILE	*fs;//File pointer
fs=fopen("spikes.dat","w");

initialize();

for(sec=0;sec<60;sec++) {//plot for 1minute(60s)
for(t=0;t<1000;t++)//plot for 1sec(1000ms)
  {
  for(i=0;i<total;i++)inputs[i]=0.0;//Fresh input
  for(k=0;k<total/1000;k++) inputs[pstochastic(total)]=20.0;//Noisy input
  int fired_count = 0;
  for(i=0;i<total;i++)
    if(v[i]>=30)//Passing the threshold:
      {
        fired_count += 1;
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
    if (t%50==0) fprintf(fs,"%d\n", fired_count);

  }

std::cout<<"sec="<<sec<<",firingrate="<<float(num_fired)/total<<"\n";
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
fclose(fs);
return 1;
}
