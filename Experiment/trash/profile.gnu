set term postscript eps color enhanced
set output "profiling_4_qubits.eps"

set logscale y

p './profiling_4' index 0 u 0:10 w lp title "V matrix, comm", \
  './profiling_4' index 0 u 0:11 w lp title "V matrix, map",  \
  './profiling_4' index 0 u 0:12 w lp title "W matrix, comm", \
  './profiling_4' index 0 u 0:13 w lp title "W matrix, map",  \
  './profiling_4' index 1 u 0:10 w lp title "M matrix, comm", \
  './profiling_4' index 1 u 0:11 w lp title "M matrix, map",  \
  './profiling_4' index 1 u 0:12 w lp title "Q matrix, comm", \
  './profiling_4' index 1 u 0:13 w lp title "Q matrix, map"

set term postscript eps color enhanced
set output "profiling_6_qubits.eps"

set logscale y

p './profiling_6' index 0 u 0:10 w lp title "V matrix, comm", \
  './profiling_6' index 0 u 0:11 w lp title "V matrix, map",  \
  './profiling_6' index 0 u 0:12 w lp title "W matrix, comm", \
  './profiling_6' index 0 u 0:13 w lp title "W matrix, map",  \
  './profiling_6' index 1 u 0:10 w lp title "M matrix, comm", \
  './profiling_6' index 1 u 0:11 w lp title "M matrix, map",  \
  './profiling_6' index 1 u 0:12 w lp title "Q matrix, comm", \
  './profiling_6' index 1 u 0:13 w lp title "Q matrix, map"

