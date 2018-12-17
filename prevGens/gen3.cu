//Editor: Michael Lukiman
//Izhikevich spiking neuron network implementation in CUDA with added spatial winner-take-all dynamics
//GPU Architecture and Programming - Fall 2018

#include <stdio.h> //Standard input-output
#include <stdlib.h> //StandardLibraryfunctions
#include <iostream> //For streaming input-output operations
#include <math.h> //math

//Define parameters of the network:
const int excitatory=256;//Excitatory neurons(N_e)
const int inhibitory=256;//Inhibitory neurons(N_i)
const int total=excitatory+inhibitory;//Total Exc.+Inh. neurons
const int synapses=100;//synapses per neuron
const int delay=20;//in milliseconds, top axonal conduction delay
const int hz=total*120;//Upper bound of firingrate
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
float LTpot[total][delay];//Longterm potentiation
float LTdep[total];//Longterm depression

//Turn these relationships into data arrays
int ps_set[total][synapses];//Matrix holding the post-synaptic neurons(synapses) of each neuron(total)
float weights[total][synapses];//Matrix holding the weights of each synapse
float w_derivs[total][synapses];//Matrix holding the derivative of each above weight
int delays_length[total][delay];//Matrix holding the delay values of each neuron
int del_set[total][delay][synapses];//Matrix holding the delays to each synapse from each neuron
int pre_neuron[total];//Index of presynaptic information
int pre_input[total][synapses*3];//Presynaptic inputs
int pre_delay[total][synapses*3];//Presynaptic delays
float *pre_weights[total][synapses*3];//Presynaptic weights
float *pre_w_derivs[total][synapses*3];//Presynaptic derivatives

int pstochastic(int n) { // Pseudo-stochastic/random
  return rand() % (int)(n);
}

void initialize()
{	int i,j,k,jj,dd, exists, r;
	for (i=0;i<excitatory;i++) a[i]=0.02;// RS type
	for (i=excitatory;i<total;i++) a[i]=0.1; // FS type
	for (i=0;i<excitatory;i++) d[i]=8.0; // RS type
	for (i=excitatory;i<total;i++) d[i]=2.0; // FS type

	for (i=0;i<total;i++) for (j=0;j<synapses;j++)
	{
		do{
			exists = 0;		// avoid multiple synapses
			if (i<excitatory) r = pstochastic(total);
			else	  r = pstochastic(excitatory);// inh -> exc only
			if (r==i) exists=1;									// no self-synapses
			for (k=0;k<j;k++) if (ps_set[i][k]==r) exists = 1;	// synapse already exists
		}while (exists == 1);
		ps_set[i][j]=r;
	}
	for (i=0;i<excitatory;i++)	for (j=0;j<synapses;j++) weights[i][j]=6.0;  // initial exc. synaptic weights
	for (i=excitatory;i<total;i++)	for (j=0;j<synapses;j++) weights[i][j]=-5.0; // inhibitory synaptic weights
  	for (i=0;i<total;i++)	for (j=0;j<synapses;j++) w_derivs[i][j]=0.0; // synaptic derivatives
  	for (i=0;i<total;i++)
	{
		short ind=0;
		if (i<excitatory)
		{
			for (j=0;j<delay;j++)
			{	delays_length[i][j]=synapses/delay;	// uniform distribution of exc. synaptic delays
				for (k=0;k<delays_length[i][j];k++)
					del_set[i][j][k]=ind++;
			}
		}
		else
		{
			for (j=0;j<delay;j++) delays_length[i][j]=0;
			delays_length[i][0]=synapses;			// all inhibitory delays are 1 ms
			for (k=0;k<delays_length[i][0];k++)
					del_set[i][0][k]=ind++;
		}
	}

  	for (i=0;i<total;i++)
	{
		pre_neuron[i]=0;
		for (j=0;j<excitatory;j++)
		for (k=0;k<synapses;k++)
		if (ps_set[j][k] == i)		// find all presynaptic neurons
		{
			pre_input[i][pre_neuron[i]]=j;	// add this neuron to the list
			for (dd=0;dd<delay;dd++)	// find the delay
				for (jj=0;jj<delays_length[j][dd];jj++)
					if (ps_set[j][del_set[j][dd][jj]]==i) pre_delay[i][pre_neuron[i]]=dd;
			pre_weights[i][pre_neuron[i]]=&weights[j][k];	// pointer to the synaptic weight
			pre_w_derivs[i][pre_neuron[i]++]=&w_derivs[j][k];// pointer to the derivative
		}
	}

	for (i=0;i<total;i++)	for (j=0;j<1+delay;j++) LTpot[i][j]=0.0;
	for (i=0;i<total;i++)	LTdep[i]=0.0;
	for (i=0;i<total;i++)	v[i]=-65.0;		// initial values for v
	for (i=0;i<total;i++)	u[i]=0.2*v[i];	// initial values for u

	num_fired=1;		// spike timings
	spike[0][0]=-delay;	// put a dummy spike at -delay for simulation efficiency
	spike[0][1]=0;	// index of the dummy spike
}

int main()
{
	int		i, j, k, sec, t;
	float	input[total];
	FILE	*fs;

	initialize();	// assign connections, weights, etc.

	for (sec=0; sec<60; sec++)		// simulation of 1 day
	{
		for (t=0;t<1000;t++)				// simulation of 1 sec
		{
			for (i=0;i<total;i++) input[i] = 0.0;	// reset the input
			for (k=0;k<total/1000;k++)
				input[pstochastic(total)]=20.0;		// random thalamic input
			for (i=0;i<total;i++)
			if (v[i]>=30)					// did it fire?
			{
				v[i] = -65.0;					// voltage reset
				u[i]+=d[i];					// recovery variable reset
				LTpot[i][t+delay]= 0.1;
				LTdep[i]=0.12;
				for (j=0;j<pre_neuron[i];j++) *pre_w_derivs[i][j]+=LTpot[pre_input[i][j]][t+delay-pre_delay[i][j]-1];// this spike was after pre-synaptic spikes
				spike[num_fired  ][0]=t;
				spike[num_fired++][1]=i;
				if (num_fired == hz) {std::cout << "Two many spikes at t=" << t << " (ignoring all)";num_fired=1;}
			}
			k=num_fired;
			while (t-spike[--k][0] <delay)
			{
				for (j=0; j< delays_length[spike[k][1]][t-spike[k][0]]; j++)
				{
					i=ps_set[spike[k][1]][del_set[spike[k][1]][t-spike[k][0]][j]];
					input[i]+=weights[spike[k][1]][del_set[spike[k][1]][t-spike[k][0]][j]];
					if (spike[k][1] <excitatory) // this spike is before postsynaptic spikes
						w_derivs[spike[k][1]][del_set[spike[k][1]][t-spike[k][0]][j]]-=LTdep[i];
				}
			}
			for (i=0;i<total;i++)
			{
				v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+input[i]); // for numerical stability
				v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+input[i]); // time step is 0.5 ms
				u[i]+=a[i]*(0.2*v[i]-u[i]);
				LTpot[i][t+delay+1]=0.95*LTpot[i][t+delay];
				LTdep[i]*=0.95;
			}
		}
		std::cout << "sec=" << sec << ", firing rate=" << float(num_fired)/total << "\n";
   		fs = fopen("spikes.dat","w+");
      printf("%d", num_fired);
		for (i=1;i<num_fired;i++)
			if (spike[i][0] >=0)
				fprintf(fs, "%d  %d\nfffff", spike[i][0], spike[i][1]);
		fclose(fs);

		for (i=0;i<total;i++)		// prepare for the next sec
			for (j=0;j<delay+1;j++)
			LTpot[i][j]=LTpot[i][1000+j];
		k=num_fired-1;
		while (1000-spike[k][0]<delay) k--;
		for (i=1;i<num_fired-k;i++)
		{
			spike[i][0]=spike[k+i][0]-1000;
			spike[i][1]=spike[k+i][1];
		}
		num_fired = num_fired-k;

		for (i=0;i<excitatory;i++)	// modify only exc connections
		for (j=0;j<synapses;j++)
		{
			weights[i][j]+=0.01+w_derivs[i][j];
			w_derivs[i][j]*=0.9;
			if (weights[i][j]>max_weight) weights[i][j]=max_weight;
			if (weights[i][j]<0) weights[i][j]=0.0;
		}
	}
}
