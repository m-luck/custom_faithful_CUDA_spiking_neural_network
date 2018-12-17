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

__global__ void millisecondUpdate(float * devI,
																	int * devpost,
																	float * devs,
																	float * devsd,
																	short * devdelays_length,
																	short * devdelays,
																	int * devN_pre,
																	int * devI_pre,
																	int * devD_pre,
																	float * devs_pre,
																	float * devsd_pre,
																	float * devLTP,
																	float * devLTD,
																	float * deva,
																	float * devd,
																	float * devv,
																	float * devu,
																	int * devfirings,
																	int * devN_firings,
																	int t)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < devN) {
		if (devv[i]>=30)					// did it fire?
		{
			devv[i] = -65.0;					// voltage reset
			devu[i]+=devd[i];					// recovery variable reset
			devLTP[i*(1000+D)+t+D]= 0.1;
			devLTD[i]=0.12;
			for (int j=0;j<devN_pre[i];j++) devsd_pre[ i*(3*M)+j ] += devLTP[ ( devI_pre [i*(3*M)+j] ) * (1001+D) + t+D-devD_pre[i*3*M+j]-1 ];// this spike was after pre-synaptic spikes
			devfirings[*devN_firings * 2]=t;
			devfirings[*devN_firings++ * 2 + 1]=i;
			if (*devN_firings == devN_firings_max) { printf("Two many spikes at t=%d (ignoring all)",t); *devN_firings=1;}
		}
		int k=*devN_firings;
		while (t-devfirings[(--k)*2] <D)
		{
			for (int j=0; j< devdelays_length[devfirings[k*2+1] * D + t-devfirings[k*2]]; j++)
			{
				i=devpost[ devfirings[k*2+1] * M + devdelays[devfirings[k*2+1] * D + (t-devfirings[k*2] * M + j)]];
				devI[i]+=devs[devfirings[k*2+1] * M + devdelays[devfirings[k*2+1] * D + (t-devfirings[k*0] * M + j)]];
				if (devfirings[k*2+1] <Ne) // this spike is before postsynaptic spikes
					devsd[devfirings[k*2+1] * M + devdelays[devfirings[k*2+1] * D + (t-devfirings[k*2] *M + j)]]-=devLTD[i];
			}
		}
		for (int i=0;i<N;i++)
		{
			devv[i]+=0.5*((0.04*devv[i]+5)*devv[i]+140-devu[i]+devI[i]); // for numerical stability
			devv[i]+=0.5*((0.04*devv[i]+5)*devv[i]+140-devu[i]+devI[i]); // time step is 0.5 ms
			devu[i]+=deva[i]*(0.2*devv[i]-devu[i]);
			devLTP[i * (1001+D) + t+D+1]=0.95*devLTP[i * (1001+D) + t+D];
			devLTD[i]*=0.95;
		}
	}
}

int main()
{
	int		i, j, k, sec, t;
	FILE	*fs;
	initialize(devpost);	// assign connections, weights, etc.

// Neurons take a lot of matrices...time for the most cudaMalloc'ing ever done by me.
	cudaMalloc(&devpost, sizeof(int)*N*M);
	cudaMalloc(&devs, sizeof(int)*N*M);
	cudaMalloc(&devsd, sizeof(int)*N*M);
	cudaMalloc(&devdelays_length, sizeof(short)*N*D);
	cudaMalloc(&devdelays, sizeof(short)*N*D*M);
	cudaMalloc(&devN_pre, sizeof(int)*N);
	cudaMalloc(&devI_pre, sizeof(int)*N*3*M);
	cudaMalloc(&devD_pre, sizeof(int)*N*3*M);
	cudaMalloc(&devs_pre, sizeof(float*)*N*3*M);
	cudaMalloc(&devsd_pre, sizeof(float*)*N*3*M);
	cudaMalloc(&devLTP, sizeof(float)*N*(1001+D));
	cudaMalloc(&devLTD, sizeof(float)*N);
	cudaMalloc(&deva, sizeof(float)*N);
	cudaMalloc(&devd, sizeof(float)*N);
	cudaMalloc(&devv, sizeof(float)*N);
	cudaMalloc(&devu, sizeof(float)*N);
	cudaMalloc(&devfirings, sizeof(int)*N_firings_max*2);

	for (sec=0; sec<60*60*24; sec++)		// simulation of 1 day
	{
		for (t=0;t<1000;t++)				// simulation of 1 sec
		{
			for (i=0;i<N;i++) I[i] = 0.0;	// reset the input
			for (k=0;k<N/1000;k++)
				I[getrandom(N)]=20.0;		// random thalamic input

			cudaMemcpy(devN_firings, &N_firings, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(devI, I, sizeof(float)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(devpost, post, sizeof(int)*N*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devs, s, sizeof(float)*N*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devsd, d, sizeof(float)*N*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devdelays_length, delays_length, sizeof(short)*N*D, cudaMemcpyHostToDevice);
			cudaMemcpy(devdelays, delays, sizeof(short)*N*D*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devN_pre, N_pre, sizeof(int)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(devI_pre, I_pre, sizeof(int)*N*3*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devD_pre, D_pre, sizeof(int)*N*3*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devs_pre, s_pre, sizeof(float*)*N*3*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devsd_pre, sd_pre, sizeof(float*)*N*3*M, cudaMemcpyHostToDevice);
			cudaMemcpy(devLTP, LTP, sizeof(float)*N*(1001+D), cudaMemcpyHostToDevice);
			cudaMemcpy(devLTD, LTD, sizeof(float)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(deva, a, sizeof(float)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(devd, d, sizeof(float)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(devv, v, sizeof(float)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(devu, u, sizeof(float)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(devfirings, firings, sizeof(int)*N_firings_max*2, cudaMemcpyHostToDevice);
			millisecondUpdate<<<7,128>>>(devI,
																	devpost,
																	devs,
																	devsd,
																	devdelays_length,
																	devdelays,
																	devN_pre,
																	devI_pre,
																	devD_pre,
																	devs_pre,
																	devsd_pre,
																	devLTP,
				 													devLTD,
				  												deva,
																	devd,
																	devv,
																	devu,
																	devfirings,
																	devN_firings,
																	t);
			cudaMemcpy(devfirings, firings, sizeof(int)*N_firings_max*2, cudaMemcpyDeviceToHost);
			cudaMemcpy(devu, u, sizeof(float)*N, cudaMemcpyDeviceToHost);
			cudaMemcpy(v, devv,  sizeof(float)*N, cudaMemcpyDeviceToHost);
			cudaMemcpy(d, devd, sizeof(float)*N, cudaMemcpyDeviceToHost);
			cudaMemcpy(a, deva, sizeof(float)*N, cudaMemcpyDeviceToHost);
			cudaMemcpy(LTD, devLTD, sizeof(float)*N, cudaMemcpyDeviceToHost);
			cudaMemcpy(LTP, devLTP, sizeof(float)*N*(1001+D), cudaMemcpyDeviceToHost);
			cudaMemcpy(sd_pre, devsd_pre, sizeof(float*)*N*3*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(s_pre, devs_pre, sizeof(float*)*N*3*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(D_pre, devD_pre,  sizeof(int)*N*3*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(I_pre, devI_pre, sizeof(int)*N*3*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(N_pre, devN_pre, sizeof(int)*N, cudaMemcpyDeviceToHost);
			cudaMemcpy(delays, devdelays, sizeof(short)*N*D*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(delays_length, devdelays_length, sizeof(short)*N*D, cudaMemcpyDeviceToHost);
			cudaMemcpy(sd, devsd, sizeof(float)*N*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(s, devs, sizeof(float)*N*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(post, devpost, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
			cudaMemcpy(I, devI, sizeof(float)*N, cudaMemcpyDeviceToHost);
			cudaMemcpy(&N_firings, devN_firings, sizeof(int), cudaMemcpyDeviceToHost);
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
