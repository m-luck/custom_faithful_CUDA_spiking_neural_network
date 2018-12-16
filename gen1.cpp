//Editor: Michael Lukiman
//Izhikevich spiking neuron network implementation in CUDA with added spatial winner-take-all dynamics
//GPU Architecture and Programming - Fall 2018

#include <stdio.h> //Standard input-output
#include <stlib.h> //StandardLibraryfunctions
#include <iostream.h> //For streaming input-output operations
#include <math.h> //math

//Unsigned integers for always positive numbers
#typedef unsigned int uint

//Define parameters of the network:
const uint excitatory=256;//Excitatory neurons(N_e)
const uint inhibitory=256;//Inhibitory neurons(N_i)
const uint total=excitatory+inhibitory;//Total Exc.+Inh. neurons
const uint syn_per=100;//synapses per neuron
const uint delay=20;//in milliseconds, top axonal conduction delay
const uint hz=total*120;//Upper bound of firingrate
const float synweight=10.0;//Top synaptic weight

//Neuronal dynamics
float a[total];
float d[total];

//Activity variables
float v[total];
float u[total];

uint num_fired;//amount of fired neurons
uint spike_times[hz][2];//Timing of spikes, max of hz limit

//Spike-timing dependent variables (LTP,LTD)
float LTpot[total][delay];//Longterm potentiation
float LTdep[total];//Longterm depression

//Turn these relationships into data arrays
uint ps_set[total][syn_per];//Matrix holding the post-synaptic neurons(syn_per) of each neuron(total)
float weights[total][syn_per];//Matrix holding the weights of each synapse
float w_derivs[total][syn_per];//Matrix holding the derivative of each above weight
short delay_length[total][delay];//Matrix holding the delay values of each neuron
short delays[total][delay][syn_per];//Matrix holding the delays to each synapse from each neuron
uint pre_neuron[total];//Index of presynaptic information
uint pre_i[total][syn_per*3];//Presynaptic inputs
uint pre_delay[total][syn_per*3];//Presynaptic delays
float *pre_weights[total][syn_per*3];//Presynaptic weights
float *pre_w_derivs[total][syn_per*3];//Presynaptic derivatives

voidinitialize()
{
uint i,j,k;
uint j_j;
uint d_d;
uint init;
uint r;

for(i=0;i<excitatory;i++)a[i]=0.02//Set excitatory as regular-spiking neurons

for(i=excitatory;i<N;i++)a[i]=0.1;//Set inhibitory as fast-spiking neurons

for(i=0;i<excitatory;i++)d[i]=8.0//Set excitatory as regular-spiking neurons

for(i=excitatory;i<N;i++)d[i]=2.0;//Set inhibitory as fast-spiking neurons

//Self-sort synapses:
for(i=0;i<N;i++)//Every neuron
for(j=0;j<syn_per;j++)//Every weight of connected synapse
  {
  do{
  init=0;

  if(i<excitatory)r=pstochastic(total);//Pick a random neuron
  elser=pstochastic(excitatory);//Pick a random excitatory neurons

  if(r==i)self=1;//Self selection

  //Not a recurrent network, so also prevent synapses connecting to self
  for(k=0;k<j;k++)
  if(ps_set[i][k]==r)//If the synapse connects to self
  self=1;

  }
  while(self==1);

ps_set[i][j]=r//This synapse is randomly assigned
}

//Initialize excitatory synaptic weights
for(i=0;i<excitatory;i++)
for(j=0;j<syn_per;j++)
weights[i][j]=6.0;

//Initialize inhibitory weights
for(i=excitatory;i<total;i++)
for(j=0;j<syn_per;j++)
weights[i][j]=-5.0;

//Initialize synaptic derivatives
for(i=0;i<total;i++)
for(j=0;j<syn_per;j++)
w_derivs[i][j]=0.0;

for(i=0;i<N;i++)//For every neuron
  {
  shortind=0;//Keep track of its index

  if(i<excitatory)//If the neuron is excitatory
    {
    //Update delay lengths
    for(j=0;j<delays;j++)
    {
    delay_length[i][j]=syn_per/delay;//Allocate equal intervals
      //Update delays via delay lengths
      for(k=0;k<delay_length[i][j];k++)
      delays[i][j][k]=ind++;
    }
    }

  else
    {
    //Set all delays to 1ms from delay start
    for(j=0;j<delays;j++)
    delay_length[i][j]=0;

    delay_length[i][0]=syn_per;

    for(k=0;k<delay_length[i][0];k++)
    delays[i][0][k]=ind;

    ind++;
    }
}

for(i=0;i<N;i++)
  {
  pre_neuron[i]=0;
  for(j=0;j<excitatory;j++)
  for(k=0;k<syn_per;k++)
    if(ps_set[j][k]==i)//This is a presynaptic neuron
      {
      pre_i[i][pre_neuron[i]]=j;//Register it to the set

      for(d_d=0;d_d<delay;d_d++)//Every delay
      for(j_j=0;j_j<delay_length[j][d_d];j_j++)
      if(ps_set[j][delay_length[j][d_d][j_j]]==i)
        pre_delay[i][pre_neuron[i]]=d_d
      pre_weights[i][pre_neuron[i]]=&weight[j][k];//Presynaptic weight assigned to relevant synaptic weights
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
fired[0][0]=-D;//dummy spike with negative delay interval for warmup
fired[0][1]=0;//dummy spike

}

void main()
{

uinti,j,k;//Loop counters
uintsec,t;//seconds, milliseconds
float	inputs[total];//Inputs to neurons!
FILE	*fs;//File pointer

initialize();

for(sec=0;sec<60;sec++)//plot for 1minute(60s)
for(t=0;t<1000;t++)//plot for 1sec(1000ms)
  {
  for(i=0;i<total;i++)inputs[i]=0.0;//Fresh input
  for(k=0;k<total/1000;k++) inputs[pstochastic(total)]=20.0;//Noisy input

  for(i=0;i<total;i++)
    if(v[i]>=30)//Passing the threshold:
      {
      v[i]=-65.0;//Zero the voltage
      u[i]+=d[i];//Refractory period
      LTpot[i][t+D]=0.1;// Update potentiation
      LTdep[i]=0.12;//Update depression

      for(j=0;j<pre_neuron[i];j++)
        *pre_w_derivs[i][j]+=LTpot[pre_i[i][j]][t+delay-pre_delay[i][j]-1];//Spike after presynaptic spike
      fired[num_fired][0]=t;
      fired[num_fired++][1]=i;
      if(num_fired==hz){ cout<<"Too many spikes at t="<<t<<"(ignoring all)"; num_fired=1; }
      }

  k=num_fired;
  while(t-fired[--k][0]<delay)
    {
    for(j=0;j<delay_length[fired[k][1]][t-fired[k][0]];j++)
      {
      i=post[fired[k][1]][delays[fired[k][1]][t-fired[k][0]][j]];
      inputs[i]+=s[fired[k][1]][delays[fired[k][1]][t-fired[k][0]][j]];
      if(fired[k][1]<excitatory)//Spike before post-synaptic
      	w_derivs[fired[k][1]][delays[fired[k][1]][t-fired[k][0]][j]]-=LTdep[i];
      }
    }

  for(i=0;i<N;i++)
    {
    v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]);//Izkevich formulae
    v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]);//Repeat
    u[i]+=a[i]*(0.2*v[i]-u[i]);
    LTpot[i][t+D+1]=0.95*LTpot[i][t+D];
    LTdep[i]*=0.95;
    }
  }

cout<<"sec="<<sec<<",firingrate="<<float(num_fired)/N<<"\n";
fs=fopen("spikes.dat","w");
for(i=1;i<num_fired;i++)
if(fired[i][0]>=0)
  fprintf(fs,"%d%d\n",fired[i][0],fired[i][1]);
fclose(fs);

//Next second
for(i=0;i<total;i++)
for(j=0;j<delay+1;j++)
  LTpot[i][j]=LTpot[i][1000+j];

k=num_fired-1;
while(1000-fired[k][0]<delay)k--;
  for(i=1;i<num_fired-k;i++)
  {
  fired[i][0]=fired[k+i][0]-1000;
  fired[i][1]=fired[k+i][1];
  }
num_fired=num_fired-k;

for(i=0;i<Ne;i++)	//Update excitatory connections
for(j=0;j<M;j++)
  {
  s[i][j]+=0.01+sd[i][j];
  sd[i][j]*=0.9;
  if(s[i][j]>sm)s[i][j]=sm;
  if(s[i][j]<0)s[i][j]=0.0;
  }
}
