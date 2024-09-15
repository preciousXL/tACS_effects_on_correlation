COMMENT

This file is slightly adapted from Brette et al. 2007:
https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=83319&file=/destexhe_benchmarks/NEURON/coba/spikeout.mod#tabs-2

This NEURON mechanism implements IF-like spiking: it resets the membrane voltage when it reaches a threshold and fixes it for a given refractory time.

ENDCOMMENT


NEURON {
   POINT_PROCESS SpikeOut
   RANGE thresh, refrac, vrefrac, grefrac
   NONSPECIFIC_CURRENT i
}

PARAMETER {
   thresh = -55 (millivolt)
   refrac = 1.5 (ms)
   vrefrac = -65 (millivolt)
   grefrac = 100 (microsiemens) : clamp to vrefrac
}

ASSIGNED {
   i (nanoamp)
   v (millivolt)
   g (microsiemens)
}

INITIAL {
   net_send(0, 3)
   g = 0
}

: up to here, executed once

BREAKPOINT {
   i = g*(v - vrefrac)
}

NET_RECEIVE(w) {
   if (flag == 1) {
      v = vrefrac
      g = grefrac
      net_event(t) : tells that an event, spike, occured at time t to other neurons.
      net_send(refrac, 2)
   }else if (flag == 2) {
      g = 0
   }else if (flag == 3) {
      WATCH (v > thresh) 1 : if v >= thresh -> net_send(0,1) -> NET_RECEIVE(w=1)  watch is being
                : executed in each time step, you :dont have to call it.
   }   
}
