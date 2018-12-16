// Michael Lukiman
// Izhikevich spiking neuron network implementation in CUDA with added spatial winner-take-all dynamics

#include <iostream.h> // For streaming input-output operations
#include <math.h> // math
#include <stdio.h> // Standard input output
#include <stlib.h> // Standard Library functions

// Unsigned integers for always positive numbers
#typedef unsigned int uint


// Define parameters of the network:

const uint excitatory = 256; // Excitatory neurons (N_e)
const uint inhibitory = 256; // Inhibitory neurons (N_i)
const uint total = excitatory + inhibitory; // Total Exc. + Inh. neurons
const uint syn_per = 100; // synapses per neuron
const uint delay = 20; // in milliseconds, top axonal conduction delay
const uint hz = total*120; // Upper bound of firing rate
const float synweight = 10.0; // Top synaptic weight


// Turn these relationships into data arrays

uint ps_set[total][syn_per]; // Matrix holding the post-synaptic neurons (syn_per) of each neuron (total)

float weights[total][syn_per]; // Matrix holding the weights of each synapse
float w_derivs[total][syn_per]; // Matrix holding the derivative of each above weight

short delay_length[total][delay]; // Matrix holding the delay values of each neuron
short delays[total][delay][syn_per]; // Matrix holding the delays to each synapse from each neuron

uint pre_neuron[total]; // Index of presynaptic information
uint pre_i[total][syn_per*3];
uint pre_delay[total][syn_per*3];

float *pre_weights[total][syn_per*3]; // Presynaptic weights
float *pre_w_derivs[total][syn_per*3];

// Spike timing dependent variables (LTP, LTD)
float LTP[total][delay]; // Long term potentiation
float LTD[total]; // Long term depression

// Neuronal dynamics
float a[total];
float d[total];

// Activity variables
float v[total];
float u[total];

uint num_fired; // amount of fired neurons
uint spike_times[hz][2]; // Timing of spikes, max of hz limit

void initialize()
{
  uint i,j,k;
  uint j_j;
  uint d_d;
  uint init;
  uint r;

  for ( i = 0 ; i < excitatory; i++) a[i] = 0.02 // Set excitatory as regular spiking neurons
  for ( i = excitatory ; i < N; i++) a[i] = 0.1; // Set inhibitory as fast spiking neurons

  for ( i = 0 ; i < excitatory; i++) d[i] = 8.0 // Set excitatory as regular spiking neurons
  for ( i = excitatory ; i < N; i++) d[i] = 2.0; // Set inhibitory as fast spiking neurons

  for ( i = 0 ; i < N ; i++)
    for ( j = 0 ; j < M ; j++)
      {
        do {
          init = 0;
          if ( i < excitatory ) r = pstochastic(total)
        }
      }


}
