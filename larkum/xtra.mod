: $Id: xtra.mod,v 1.4 2014/08/18 23:15:25 ted Exp ted $


NEURON {
	SUFFIX xtra
	RANGE es : (es = max amplitude of the potential)
	RANGE Ex,Ey,Ez
	RANGE x, y, z
	GLOBAL stim : (stim = normalized waveform)
	POINTER ex 
}

PARAMETER {
	es = 0 (mV)
	x = 0 (1) : spatial coords
	y = 0 (1)
	z = 0 (1)
	Ex = 0 (1) : E-field components
	Ey = 0 (1)
	Ez = 0 (1)
}

ASSIGNED {
	v (millivolts)
	ex (millivolts)
	stim (unitless) 	
	area (micron2)
}

INITIAL {
	ex = stim*es
}


BEFORE BREAKPOINT { : before each cy' = f(y,t) setup
  ex = stim*es
}


