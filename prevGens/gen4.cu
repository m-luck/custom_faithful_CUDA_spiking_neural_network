// SPNET: Spiking neural network with axonal conduction delays and STDP
// Created by Eugene M. Izhikevich, May 17, 2004, San Diego, CA
// Saves spiking data each second in file spikes.dat
// To plot spikes, use MATLAB code: load spikes.dat;plot(spikes(:,1),spikes(:,2),'.');
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "params.h"
#define getrandom(max1) ((rand()%(int)((max1)))) // random integer between 0 and max-1



const	int		Ne = 800;		// excitatory neurons
const	int		Ni = 200;		// inhibitory neurons
const	int		N  = Ne+Ni;		// total number of neurons
const	int		M  = 100;		// the number of synapses per neuron
const	int		D  = 20;		// maximal axonal conduction delay


float	sm = 10.0;		// maximal synaptic strength
int		post[N][M];				// indeces of postsynaptic neurons
float	s[N][M], sd[N][M];		// matrix of synaptic weights and their derivatives
short	delays_length[N][D];	// distribution of delays
short	delays[N][D][M];		// arrangement of delays
int		N_pre[N], I_pre[N][3*M], D_pre[N][3*M];	// presynaptic information
float	*s_pre[N][3*M], *sd_pre[N][3*M];		// presynaptic weights
float	LTP[N][1001+D], LTD[N];	// STDP functions
float	a[N], d[N];				// neuronal dynamics parameters
float	v[N], u[N];				// activity variables
int		N_firings;				// the number of fired neurons
const int N_firings_max=100*N;	// upper limit on the number of fired neurons per sec
int		firings[N_firings_max][2]; // indeces and timings of spikes
float I[N];


int *devpost;
float	*devs, *devsd;		// matrix of synaptic weights and their derivatives
short	*devdelays_length;	// distribution of delays
short	*devdelays;		// arrangement of delays
int	*devN_pre, *devI_pre, *devD_pre;	// presynaptic information
float	*devs_pre, *devsd_pre;		// presynaptic weights
float	*devLTP, *devLTD;	// STDP functions
float	*deva, *devd;				// neuronal dynamics parameters
float	*devv, *devu;
float *devI;
int *devN_firings;
int *devfirings;

void initialize(int * devpost)
{
	int i,j,k,dd,jj,exists,r;
	for (i=0;i<Ne;i++) a[i]=0.02;// RS type
	for (i=Ne;i<N;i++) a[i]=0.1; // FS type
	for (i=0;i<Ne;i++) d[i]=8.0; // RS type
	for (i=Ne;i<N;i++) d[i]=2.0; // FS type

	for (i=0;i<N;i++) for (j=0;j<M;j++)
	{
		do{
			exists = 0;		// avoid multiple synapses
			if (i<Ne) r = getrandom(N);
			else	  r = getrandom(Ne);// inh -> exc only
			if (r==i) exists=1;									// no self-synapses
			for (k=0;k<j;k++) if (post[i][k]==r) exists = 1;	// synapse already exists
		}while (exists == 1);
		post[i][j]=r;
	}

	for (i=0;i<Ne;i++)	for (j=0;j<M;j++) s[i][j]=6.0;  // initial exc. synaptic weights
	for (i=Ne;i<N;i++)	for (j=0;j<M;j++) s[i][j]=-5.0; // inhibitory synaptic weights
  	for (i=0;i<N;i++)	for (j=0;j<M;j++) sd[i][j]=0.0; // synaptic derivatives
  	for (i=0;i<N;i++)
	{
		short ind=0;
		if (i<Ne)
		{
			for (j=0;j<D;j++)
			{	delays_length[i][j]=M/D;	// uniform distribution of exc. synaptic delays
				for (k=0;k<delays_length[i][j];k++)
					delays[i][j][k]=ind++;
			}
		}
		else
		{
			for (j=0;j<D;j++) delays_length[i][j]=0;
			delays_length[i][0]=M;			// all inhibitory delays are 1 ms
			for (k=0;k<delays_length[i][0];k++)
					delays[i][0][k]=ind++;
		}
	}

  	for (i=0;i<N;i++)
	{
		N_pre[i]=0;
		for (j=0;j<Ne;j++)
		for (k=0;k<M;k++)
		if (post[j][k] == i)		// find all presynaptic neurons
		{
			I_pre[i][N_pre[i]]=j;	// add this neuron to the list
			for (dd=0;dd<D;dd++)	// find the delay
				for (jj=0;jj<delays_length[j][dd];jj++)
					if (post[j][delays[j][dd][jj]]==i) D_pre[i][N_pre[i]]=dd;
			s_pre[i][N_pre[i]]=&s[j][k];	// pointer to the synaptic weight
			sd_pre[i][N_pre[i]++]=&sd[j][k];// pointer to the derivative
		}
	}

	for (i=0;i<N;i++)	for (j=0;j<1+D;j++) LTP[i][j]=0.0;
	for (i=0;i<N;i++)	LTD[i]=0.0;
	for (i=0;i<N;i++)	v[i]=-65.0;		// initial values for v
	for (i=0;i<N;i++)	u[i]=0.2*v[i];	// initial values for u

	N_firings=1;		// spike timings
	firings[0][0]=-D;	// put a dummy spike at -D for simulation efficiency
	firings[0][1]=0;	// index of the dummy spike
}

__constant__ int devN = N;
__constant__ int devM = M;
__constant__ int devD = D;
__constant__ int devN_firings_max = N_firings_max;
//
// __global__ void millisecondUpdate(float * devI,
// 																	int * devpost,
// 																	float * devs,
// 																	float * devsd,
// 																	short * devdelays_length,
// 																	short * devdelays,
// 																	int * devN_pre,
// 																	int * devI_pre,
// 																	int * devD_pre,
// 																	float * devs_pre,
// 																	float * devsd_pre,
// 																	float * devLTP,
// 																	float * devLTD,
// 																	float * deva,
// 																	float * devd,
// 																	float * devv,
// 																	float * devu,
// 																	int * devfirings,
// 																	int * devN_firings,
// 																	int t,
// 																	int N, int M, int D, int N_firings_max)
// {
//
// 	int i = threadIdx.x + blockIdx.x * blockDim.x;
// }

int main()
{
	int		i, j, k, sec, t;
	FILE	*fs;
	initialize(devpost);	// assign connections, weights, etc.

// Neurons take a lot of matrices...time for the most cudaMalloc'ing ever done by me.
	// cudaMalloc(&devpost, sizeof(int)*N*M);
	// cudaMalloc(&devs, sizeof(int)*N*M);
	// cudaMalloc(&devsd, sizeof(int)*N*M);
	// cudaMalloc(&devdelays_length, sizeof(short)*N*D);
	// cudaMalloc(&devdelays, sizeof(short)*N*D*M);
	// cudaMalloc(&devN_pre, sizeof(int)*N);
	// cudaMalloc(&devI_pre, sizeof(int)*N*3*M);
	// cudaMalloc(&devD_pre, sizeof(int)*N*3*M);
	// cudaMalloc(&devs_pre, sizeof(float*)*N*3*M);
	// cudaMalloc(&devsd_pre, sizeof(float*)*N*3*M);
	// cudaMalloc(&devLTP, sizeof(float)*N*(1001+D));
	// cudaMalloc(&devLTD, sizeof(float)*N);
	// cudaMalloc(&deva, sizeof(float)*N);
	// cudaMalloc(&devd, sizeof(float)*N);
	// cudaMalloc(&devv, sizeof(float)*N);
	// cudaMalloc(&devu, sizeof(float)*N);
	// cudaMalloc(&devfirings, sizeof(int)*N_firings_max*2);

	for (sec=0; sec<60*60*24; sec++)		// simulation of 1 day
	{
		for (t=0;t<1000;t++)				// simulation of 1 sec
		{
			for (i=0;i<N;i++) I[i] = 0.0;	// reset the input
			for (k=0;k<N/1000;k++)
				I[getrandom(N)]=20.0;		// random thalamic input

			for (i=0; i<N; i++)
					if (v[i]>=30)					// did it fire?
					{
						v[i] = -65.0;					// voltage reset
						u[i]+=d[i];					// recovery variable reset
						LTP[i][t+D]= 0.1;
						LTD[i]=0.12;
						for (int j=0;j<N_pre[i];j++) *sd_pre[ i*(3*M)+j ] += LTP[ ( I_pre [i][j] ) ] [ t+D-D_pre[i][j]-1 ];// this spike was after pre-synaptic spikes
						firings[N_firings][0]]=t;
						firings[N_firings++][1]=i;
						if (N_firings == N_firings_max) { printf("Two many spikes at t=%d (ignoring all)",t); N_firings=1;}
					}
			// cudaMemcpy(devN_firings, &N_firings, sizeof(int), cudaMemcpyHostToDevice);
			// cudaMemcpy(devI, I, sizeof(float)*N, cudaMemcpyHostToDevice);
			// cudaMemcpy(devpost, post, sizeof(int)*N*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devs, s, sizeof(float)*N*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devsd, d, sizeof(float)*N*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devdelays_length, delays_length, sizeof(short)*N*D, cudaMemcpyHostToDevice);
			// cudaMemcpy(devdelays, delays, sizeof(short)*N*D*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devN_pre, N_pre, sizeof(int)*N, cudaMemcpyHostToDevice);
			// cudaMemcpy(devI_pre, I_pre, sizeof(int)*N*3*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devD_pre, D_pre, sizeof(int)*N*3*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devs_pre, s_pre, sizeof(float*)*N*3*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devsd_pre, sd_pre, sizeof(float*)*N*3*M, cudaMemcpyHostToDevice);
			// cudaMemcpy(devLTP, LTP, sizeof(float)*N*(1001+D), cudaMemcpyHostToDevice);
			// cudaMemcpy(devLTD, LTD, sizeof(float)*N, cudaMemcpyHostToDevice);
			// cudaMemcpy(deva, a, sizeof(float)*N, cudaMemcpyHostToDevice);
			// cudaMemcpy(devd, d, sizeof(float)*N, cudaMemcpyHostToDevice);
			// cudaMemcpy(devv, v, sizeof(float)*N, cudaMemcpyHostToDevice);
			// cudaMemcpy(devu, u, sizeof(float)*N, cudaMemcpyHostToDevice);
			// cudaMemcpy(devfirings, firings, sizeof(int)*N_firings_max*2, cudaMemcpyHostToDevice);
			// millisecondUpdate<<<7,128>>>(devI,
			// 														devpost,
			// 														devs,
			// 														devsd,
			// 														devdelays_length,
			// 														devdelays,
			// 														devN_pre,
			// 														devI_pre,
			// 														devD_pre,
			// 														devs_pre,
			// 														devsd_pre,
			// 														devLTP,
			// 	 													devLTD,
			// 	  												deva,
			// 														devd,
			// 														devv,
			// 														devu,
			// 														devfirings,
			// 														devN_firings,
			// 														t, N, M, D, N_firings_max);
			// cudaMemcpy(devfirings, firings, sizeof(int)*N_firings_max*2, cudaMemcpyDeviceToHost);
			// cudaMemcpy(devu, u, sizeof(float)*N, cudaMemcpyDeviceToHost);
			// cudaMemcpy(v, devv,  sizeof(float)*N, cudaMemcpyDeviceToHost);
			// cudaMemcpy(d, devd, sizeof(float)*N, cudaMemcpyDeviceToHost);
			// cudaMemcpy(a, deva, sizeof(float)*N, cudaMemcpyDeviceToHost);
			// cudaMemcpy(LTD, devLTD, sizeof(float)*N, cudaMemcpyDeviceToHost);
			// cudaMemcpy(LTP, devLTP, sizeof(float)*N*(1001+D), cudaMemcpyDeviceToHost);
			// cudaMemcpy(sd_pre, devsd_pre, sizeof(float*)*N*3*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(s_pre, devs_pre, sizeof(float*)*N*3*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(D_pre, devD_pre,  sizeof(int)*N*3*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(I_pre, devI_pre, sizeof(int)*N*3*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(N_pre, devN_pre, sizeof(int)*N, cudaMemcpyDeviceToHost);
			// cudaMemcpy(delays, devdelays, sizeof(short)*N*D*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(delays_length, devdelays_length, sizeof(short)*N*D, cudaMemcpyDeviceToHost);
			// cudaMemcpy(sd, devsd, sizeof(float)*N*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(s, devs, sizeof(float)*N*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(post, devpost, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
			// cudaMemcpy(I, devI, sizeof(float)*N, cudaMemcpyDeviceToHost);
			// cudaMemcpy(&N_firings, devN_firings, sizeof(int), cudaMemcpyDeviceToHost);


			k=N_firings;
			while (t-firings[--k][0] < D)
			{
				for (j=0; j< delays_length[firings[k][1]][t-firings[k][0]]; j++)
				{
					i=post[firings[k][1]][delays[firings[k][1]][t-firings[k][0]][j]];
					I[i]+=s[firings[k][1]][delays[firings[k][1]][t-firings[k][0]][j]];
					if (firings[k][1] <Ne) // this spike is before postsynaptic spikes
						sd[firings[k][1]][delays[firings[k][1]][t-firings[k][0]][j]]-=LTD[i];
				}
			}
			for (i=0;i<N;i++)
			{
				v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]); // for numerical stability
				v[i]+=0.5*((0.04*v[i]+5)*v[i]+140-u[i]+I[i]); // time step is 0.5 ms
				u[i]+=a[i]*(0.2*v[i]-u[i]);
				LTP[i][t+D+1]=0.95*LTP[i][t+D];
				LTD[i]*=0.95;
			}
		}
		std::cout << "sec=" << sec << ", firing rate=" << float(N_firings)/N << "\n";
   		fs = fopen("spikes.dat","w");
		for (i=1;i<N_firings;i++)
			if (firings[i][0] >=0)
				fprintf(fs, "%d  %d\n", firings[i][0], firings[i][1]);
		fclose(fs);

		for (i=0;i<N;i++)		// prepare for the next sec
			for (j=0;j<D+1;j++)
			LTP[i][j]=LTP[i][1000+j];
		k=N_firings-1;
		while (1000-firings[k][0]<D) k--;
		for (i=1;i<N_firings-k;i++)
		{
			firings[i][0]=firings[k+i][0]-1000;
			firings[i][1]=firings[k+i][1];
		}
		N_firings = N_firings-k;

		for (i=0;i<Ne;i++)	// modify only exc connections
		for (j=0;j<M;j++)
		{
			s[i][j]+=0.01+sd[i][j];
			sd[i][j]*=0.9;
			if (s[i][j]>sm) s[i][j]=sm;
			if (s[i][j]<0) s[i][j]=0.0;
		}
  }
}
