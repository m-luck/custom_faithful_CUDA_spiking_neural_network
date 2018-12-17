#define getrandom(max1) ((rand()%(int)((max1))))

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
float	I[N];

int *devpost;
float	*devs, *devsd;		// matrix of synaptic weights and their derivatives
short	*devdelays_length;	// distribution of delays
short	*devdelays;		// arrangement of delays
int	*devN_pre, *devI_pre, *devD_pre;	// presynaptic information
float	*devs_pre, *devsd_pre;		// presynaptic weights
float	*devLTP, *devLTD;	// STDP functions
float	*deva, *devd;				// neuronal dynamics parameters
float	*devv, *devu;
int * devfirings;
float * devI;
int * devN_firings;

__constant__ int devN = N;
__constant__ int devM = M;
__constant__ int devD = D;
__constant__ int devN_firings_max = N_firings_max;

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
