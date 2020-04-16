set term postscript eps color enhanced
set output "timings_4_ovlp.eps"

set logscale y

p './timings' index 12 u 0:9  w lp title "COBRA, V",       \
  './timings' index 12 u 0:10 w lp title "COBRA, V (map)", \
  './timings' index 12 u 0:11 w lp title "COBRA, W",       \
  './timings' index 12 u 0:12 w lp title "COBRA, W (map)"

set term postscript eps color enhanced
set output "timings_4_hmat.eps"

set logscale y

p './timings' index 13 u 0:9  w lp title "COBRA, Q",       \
  './timings' index 13 u 0:10 w lp title "COBRA, Q (map)", \
  './timings' index 13 u 0:11 w lp title "COBRA, M",       \
  './timings' index 13 u 0:12 w lp title "COBRA, M (map)"

# =====

set term postscript eps color enhanced
set output "timings_4_ovlp_a.eps"

set logscale y

p './timings' index 14 u 0:9  w lp title "Math, V",       \
  './timings' index 14 u 0:10 w lp title "Math, V (map)", \
  './timings' index 14 u 0:11 w lp title "Math, W",       \
  './timings' index 14 u 0:12 w lp title "Math, W (map)"

set term postscript eps color enhanced
set output "timings_4_hmat_a.eps"

set logscale y

p './timings' index 15 u 0:9  w lp title "Math, Q",       \
  './timings' index 15 u 0:10 w lp title "Math, Q (map)", \
  './timings' index 15 u 0:11 w lp title "Math, M",       \
  './timings' index 15 u 0:12 w lp title "Math, M (map)"

