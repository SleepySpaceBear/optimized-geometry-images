using Dates
using SparseArrays
using Images 
using FileIO

include("obj.jl")

const EPS = 2e-10


# finds the bilinear interpolation matrix for the trapezoid 
# NOTE: u0, u1, v00, v01, v10, and v11 are all on [0,1] 
# u0  - the position of the left boundary of the trapezoid 
# u1  - the position of the right boundary of the trapezoid 
# v00 - the height of the bottom left vertex of the trapezoid 
# v01 - the height of the top left vertex of the trapezoid
# v10 - the height of the bottom right vertex of the trapezoid 
# v11 - the height of the top right vertex of the trapezoid
function find_trapezoid_bi_value(u0::Float64, u1::Float64, v00::Float64,
		v01::Float64, v10::Float64, v11::Float64) 
	# this math comes from ogi_gnarly_integrals.py
	Q11 = -u0^2*v00^3/18 - u0^2*v00^2*v10/30 + u0^2*v00^2/5 -
		u0^2*v00*v10^2/60 + u0^2*v00*v10/10 - u0^2*v00/4 + u0^2*v01^3/18 +
		u0^2*v01^2*v11/30 - u0^2*v01^2/5 + u0^2*v01*v11^2/60 -
		u0^2*v01*v11/10 + u0^2*v01/4 - u0^2*v10^3/180 + u0^2*v10^2/30 -
		u0^2*v10/12 + u0^2*v11^3/180 - u0^2*v11^2/30 + u0^2*v11/12 -
		u0*u1*v00^3/45 - u0*u1*v00^2*v10/30 + u0*u1*v00^2/10 -
		u0*u1*v00*v10^2/30 + 2*u0*u1*v00*v10/15 - u0*u1*v00/6 + u0*u1*v01^3/45 +
		u0*u1*v01^2*v11/30 - u0*u1*v01^2/10 + u0*u1*v01*v11^2/30 -
		2*u0*u1*v01*v11/15 + u0*u1*v01/6 - u0*u1*v10^3/45 + u0*u1*v10^2/10 -
		u0*u1*v10/6 + u0*u1*v11^3/45 - u0*u1*v11^2/10 + u0*u1*v11/6 +
		2*u0*v00^3/15 + u0*v00^2*v10/10 - u0*v00^2/2 + u0*v00*v10^2/15 -
		u0*v00*v10/3 + 2*u0*v00/3 - 2*u0*v01^3/15 - u0*v01^2*v11/10 +
		u0*v01^2/2 - u0*v01*v11^2/15 + u0*v01*v11/3 - 2*u0*v01/3 + u0*v10^3/30 -
		u0*v10^2/6 + u0*v10/3 - u0*v11^3/30 + u0*v11^2/6 - u0*v11/3 -
		u1^2*v00^3/180 - u1^2*v00^2*v10/60 + u1^2*v00^2/30 -
		u1^2*v00*v10^2/30 + u1^2*v00*v10/10 - u1^2*v00/12 + u1^2*v01^3/180 +
		u1^2*v01^2*v11/60 - u1^2*v01^2/30 + u1^2*v01*v11^2/30 -
		u1^2*v01*v11/10 + u1^2*v01/12 - u1^2*v10^3/18 + u1^2*v10^2/5 -
		u1^2*v10/4 + u1^2*v11^3/18 - u1^2*v11^2/5 + u1^2*v11/4 +
		u1*v00^3/30 + u1*v00^2*v10/15 - u1*v00^2/6 + u1*v00*v10^2/10 -
		u1*v00*v10/3 + u1*v00/3 - u1*v01^3/30 - u1*v01^2*v11/15 + u1*v01^2/6 -
		u1*v01*v11^2/10 + u1*v01*v11/3 - u1*v01/3 + 2*u1*v10^3/15 - u1*v10^2/2 +
		2*u1*v10/3 - 2*u1*v11^3/15 + u1*v11^2/2 - 2*u1*v11/3 - v00^3/12 -
		v00^2*v10/12 + v00^2/3 - v00*v10^2/12 + v00*v10/3 - v00/2 + v01^3/12 +
		v01^2*v11/12 - v01^2/3 + v01*v11^2/12 - v01*v11/3 + v01/2 - v10^3/12 +
		v10^2/3 - v10/2 + v11^3/12 - v11^2/3 + v11/2
	
	Q12 = u0^2*v00^3/18 + u0^2*v00^2*v10/30 - u0^2*v00^2/10 +
		u0^2*v00*v10^2/60 - u0^2*v00*v10/20 - u0^2*v01^3/18 -
		u0^2*v01^2*v11/30 + u0^2*v01^2/10 - u0^2*v01*v11^2/60 +
		u0^2*v01*v11/20 + u0^2*v10^3/180 - u0^2*v10^2/60 - u0^2*v11^3/180 +
		u0^2*v11^2/60 + u0*u1*v00^3/45 + u0*u1*v00^2*v10/30 - u0*u1*v00^2/20 +
		u0*u1*v00*v10^2/30 - u0*u1*v00*v10/15 - u0*u1*v01^3/45 -
		u0*u1*v01^2*v11/30 + u0*u1*v01^2/20 - u0*u1*v01*v11^2/30 +
		u0*u1*v01*v11/15 + u0*u1*v10^3/45 - u0*u1*v10^2/20 - u0*u1*v11^3/45 +
		u0*u1*v11^2/20 - 2*u0*v00^3/15 - u0*v00^2*v10/10 + u0*v00^2/4 -
		u0*v00*v10^2/15 + u0*v00*v10/6 + 2*u0*v01^3/15 + u0*v01^2*v11/10 -
		u0*v01^2/4 + u0*v01*v11^2/15 - u0*v01*v11/6 - u0*v10^3/30 +
		u0*v10^2/12 + u0*v11^3/30 - u0*v11^2/12 + u1^2*v00^3/180 +
		u1^2*v00^2*v10/60 - u1^2*v00^2/60 + u1^2*v00*v10^2/30 -
		u1^2*v00*v10/20 - u1^2*v01^3/180 - u1^2*v01^2*v11/60 +
		u1^2*v01^2/60 - u1^2*v01*v11^2/30 + u1^2*v01*v11/20 + u1^2*v10^3/18 -
		u1^2*v10^2/10 - u1^2*v11^3/18 + u1^2*v11^2/10 - u1*v00^3/30 -
		u1*v00^2*v10/15 + u1*v00^2/12 - u1*v00*v10^2/10 + u1*v00*v10/6 +
		u1*v01^3/30 + u1*v01^2*v11/15 - u1*v01^2/12 + u1*v01*v11^2/10 -
		u1*v01*v11/6 - 2*u1*v10^3/15 + u1*v10^2/4 + 2*u1*v11^3/15 - u1*v11^2/4 +
		v00^3/12 + v00^2*v10/12 - v00^2/6 + v00*v10^2/12 - v00*v10/6 -
		v01^3/12 - v01^2*v11/12 + v01^2/6 - v01*v11^2/12 + v01*v11/6 +
		v10^3/12 - v10^2/6 - v11^3/12 + v11^2/6

	Q13 = u0^2*v00^3/18 + u0^2*v00^2*v10/30 - u0^2*v00^2/10 +
		u0^2*v00*v10^2/60 - u0^2*v00*v10/20 - u0^2*v01^3/18 -
		u0^2*v01^2*v11/30 + u0^2*v01^2/10 - u0^2*v01*v11^2/60 +
		u0^2*v01*v11/20 + u0^2*v10^3/180 - u0^2*v10^2/60 - u0^2*v11^3/180 +
		u0^2*v11^2/60 + u0*u1*v00^3/45 + u0*u1*v00^2*v10/30 - u0*u1*v00^2/20 +
		u0*u1*v00*v10^2/30 - u0*u1*v00*v10/15 - u0*u1*v01^3/45 -
		u0*u1*v01^2*v11/30 + u0*u1*v01^2/20 - u0*u1*v01*v11^2/30 +
		u0*u1*v01*v11/15 + u0*u1*v10^3/45 - u0*u1*v10^2/20 - u0*u1*v11^3/45 +
		u0*u1*v11^2/20 - 2*u0*v00^3/15 - u0*v00^2*v10/10 + u0*v00^2/4 -
		u0*v00*v10^2/15 + u0*v00*v10/6 + 2*u0*v01^3/15 + u0*v01^2*v11/10 -
		u0*v01^2/4 + u0*v01*v11^2/15 - u0*v01*v11/6 - u0*v10^3/30 +
		u0*v10^2/12 + u0*v11^3/30 - u0*v11^2/12 + u1^2*v00^3/180 +
		u1^2*v00^2*v10/60 - u1^2*v00^2/60 + u1^2*v00*v10^2/30 -
		u1^2*v00*v10/20 - u1^2*v01^3/180 - u1^2*v01^2*v11/60 +
		u1^2*v01^2/60 - u1^2*v01*v11^2/30 + u1^2*v01*v11/20 + u1^2*v10^3/18 -
		u1^2*v10^2/10 - u1^2*v11^3/18 + u1^2*v11^2/10 - u1*v00^3/30 -
		u1*v00^2*v10/15 + u1*v00^2/12 - u1*v00*v10^2/10 + u1*v00*v10/6 +
		u1*v01^3/30 + u1*v01^2*v11/15 - u1*v01^2/12 + u1*v01*v11^2/10 -
		u1*v01*v11/6 - 2*u1*v10^3/15 + u1*v10^2/4 + 2*u1*v11^3/15 - u1*v11^2/4 +
		v00^3/12 + v00^2*v10/12 - v00^2/6 + v00*v10^2/12 - v00*v10/6 -
		v01^3/12 - v01^2*v11/12 + v01^2/6 - v01*v11^2/12 + v01*v11/6 +
		v10^3/12 - v10^2/6 - v11^3/12 + v11^2/6

	Q14 = -u0^2*v00^3/18 - u0^2*v00^2*v10/30 + u0^2*v00^2/10 -
		u0^2*v00*v10^2/60 + u0^2*v00*v10/20 + u0^2*v01^3/18 +
		u0^2*v01^2*v11/30 - u0^2*v01^2/10 + u0^2*v01*v11^2/60 -
		u0^2*v01*v11/20 - u0^2*v10^3/180 + u0^2*v10^2/60 + u0^2*v11^3/180 -
		u0^2*v11^2/60 - u0*u1*v00^3/45 - u0*u1*v00^2*v10/30 + u0*u1*v00^2/20 -
		u0*u1*v00*v10^2/30 + u0*u1*v00*v10/15 + u0*u1*v01^3/45 +
		u0*u1*v01^2*v11/30 - u0*u1*v01^2/20 + u0*u1*v01*v11^2/30 -
		u0*u1*v01*v11/15 - u0*u1*v10^3/45 + u0*u1*v10^2/20 + u0*u1*v11^3/45 -
		u0*u1*v11^2/20 + u0*v00^3/15 + u0*v00^2*v10/20 - u0*v00^2/8 +
		u0*v00*v10^2/30 - u0*v00*v10/12 - u0*v01^3/15 - u0*v01^2*v11/20 +
		u0*v01^2/8 - u0*v01*v11^2/30 + u0*v01*v11/12 + u0*v10^3/60 -
		u0*v10^2/24 - u0*v11^3/60 + u0*v11^2/24 - u1^2*v00^3/180 -
		u1^2*v00^2*v10/60 + u1^2*v00^2/60 - u1^2*v00*v10^2/30 +
		u1^2*v00*v10/20 + u1^2*v01^3/180 + u1^2*v01^2*v11/60 -
		u1^2*v01^2/60 + u1^2*v01*v11^2/30 - u1^2*v01*v11/20 - u1^2*v10^3/18 +
		u1^2*v10^2/10 + u1^2*v11^3/18 - u1^2*v11^2/10 + u1*v00^3/60 +
		u1*v00^2*v10/30 - u1*v00^2/24 + u1*v00*v10^2/20 - u1*v00*v10/12 -
		u1*v01^3/60 - u1*v01^2*v11/30 + u1*v01^2/24 - u1*v01*v11^2/20 +
		u1*v01*v11/12 + u1*v10^3/15 - u1*v10^2/8 - u1*v11^3/15 + u1*v11^2/8

	Q22 = -u0^2*v00^3/18 - u0^2*v00^2*v10/30 - u0^2*v00*v10^2/60 +
		u0^2*v01^3/18 + u0^2*v01^2*v11/30 + u0^2*v01*v11^2/60 -
		u0^2*v10^3/180 + u0^2*v11^3/180 - u0*u1*v00^3/45 -
		u0*u1*v00^2*v10/30 - u0*u1*v00*v10^2/30 + u0*u1*v01^3/45 +
		u0*u1*v01^2*v11/30 + u0*u1*v01*v11^2/30 - u0*u1*v10^3/45 +
		u0*u1*v11^3/45 + 2*u0*v00^3/15 + u0*v00^2*v10/10 + u0*v00*v10^2/15 -
		2*u0*v01^3/15 - u0*v01^2*v11/10 - u0*v01*v11^2/15 + u0*v10^3/30 -
		u0*v11^3/30 - u1^2*v00^3/180 - u1^2*v00^2*v10/60 -
		u1^2*v00*v10^2/30 + u1^2*v01^3/180 + u1^2*v01^2*v11/60 +
		u1^2*v01*v11^2/30 - u1^2*v10^3/18 + u1^2*v11^3/18 + u1*v00^3/30 +
		u1*v00^2*v10/15 + u1*v00*v10^2/10 - u1*v01^3/30 - u1*v01^2*v11/15 -
		u1*v01*v11^2/10 + 2*u1*v10^3/15 - 2*u1*v11^3/15 - v00^3/12 -
		v00^2*v10/12 - v00*v10^2/12 + v01^3/12 + v01^2*v11/12 + v01*v11^2/12 -
		v10^3/12 + v11^3/12

	Q23 = -u0^2*v00^3/18 - u0^2*v00^2*v10/30 + u0^2*v00^2/10 -
		u0^2*v00*v10^2/60 + u0^2*v00*v10/20 + u0^2*v01^3/18 +
		u0^2*v01^2*v11/30 - u0^2*v01^2/10 + u0^2*v01*v11^2/60 -
		u0^2*v01*v11/20 - u0^2*v10^3/180 + u0^2*v10^2/60 + u0^2*v11^3/180 -
		u0^2*v11^2/60 - u0*u1*v00^3/45 - u0*u1*v00^2*v10/30 + u0*u1*v00^2/20 -
		u0*u1*v00*v10^2/30 + u0*u1*v00*v10/15 + u0*u1*v01^3/45 +
		u0*u1*v01^2*v11/30 - u0*u1*v01^2/20 + u0*u1*v01*v11^2/30 -
		u0*u1*v01*v11/15 - u0*u1*v10^3/45 + u0*u1*v10^2/20 + u0*u1*v11^3/45 -
		u0*u1*v11^2/20 + u0*v00^3/15 + u0*v00^2*v10/20 - u0*v00^2/8 +
		u0*v00*v10^2/30 - u0*v00*v10/12 - u0*v01^3/15 - u0*v01^2*v11/20 +
		u0*v01^2/8 - u0*v01*v11^2/30 + u0*v01*v11/12 + u0*v10^3/60 -
		u0*v10^2/24 - u0*v11^3/60 + u0*v11^2/24 - u1^2*v00^3/180 -
		u1^2*v00^2*v10/60 + u1^2*v00^2/60 - u1^2*v00*v10^2/30 +
		u1^2*v00*v10/20 + u1^2*v01^3/180 + u1^2*v01^2*v11/60 -
		u1^2*v01^2/60 + u1^2*v01*v11^2/30 - u1^2*v01*v11/20 - u1^2*v10^3/18 +
		u1^2*v10^2/10 + u1^2*v11^3/18 - u1^2*v11^2/10 + u1*v00^3/60 +
		u1*v00^2*v10/30 - u1*v00^2/24 + u1*v00*v10^2/20 - u1*v00*v10/12 -
		u1*v01^3/60 - u1*v01^2*v11/30 + u1*v01^2/24 - u1*v01*v11^2/20 +
		u1*v01*v11/12 + u1*v10^3/15 - u1*v10^2/8 - u1*v11^3/15 + u1*v11^2/8
	
	Q24 = u0^2*v00^3/18 + u0^2*v00^2*v10/30 + u0^2*v00*v10^2/60 -
		u0^2*v01^3/18 - u0^2*v01^2*v11/30 - u0^2*v01*v11^2/60 +
		u0^2*v10^3/180 - u0^2*v11^3/180 + u0*u1*v00^3/45 +
		u0*u1*v00^2*v10/30 + u0*u1*v00*v10^2/30 - u0*u1*v01^3/45 -
		u0*u1*v01^2*v11/30 - u0*u1*v01*v11^2/30 + u0*u1*v10^3/45 -
		u0*u1*v11^3/45 - u0*v00^3/15 - u0*v00^2*v10/20 - u0*v00*v10^2/30 +
		u0*v01^3/15 + u0*v01^2*v11/20 + u0*v01*v11^2/30 - u0*v10^3/60 +
		u0*v11^3/60 + u1^2*v00^3/180 + u1^2*v00^2*v10/60 +
		u1^2*v00*v10^2/30 - u1^2*v01^3/180 - u1^2*v01^2*v11/60 -
		u1^2*v01*v11^2/30 + u1^2*v10^3/18 - u1^2*v11^3/18 - u1*v00^3/60 -
		u1*v00^2*v10/30 - u1*v00*v10^2/20 + u1*v01^3/60 + u1*v01^2*v11/30 +
		u1*v01*v11^2/20 - u1*v10^3/15 + u1*v11^3/15

	Q33 = -u0^2*v00^3/18 - u0^2*v00^2*v10/30 + u0^2*v00^2/5 -
		u0^2*v00*v10^2/60 + u0^2*v00*v10/10 - u0^2*v00/4 + u0^2*v01^3/18 +
		u0^2*v01^2*v11/30 - u0^2*v01^2/5 + u0^2*v01*v11^2/60 -
		u0^2*v01*v11/10 + u0^2*v01/4 - u0^2*v10^3/180 + u0^2*v10^2/30 -
		u0^2*v10/12 + u0^2*v11^3/180 - u0^2*v11^2/30 + u0^2*v11/12 -
		u0*u1*v00^3/45 - u0*u1*v00^2*v10/30 + u0*u1*v00^2/10 -
		u0*u1*v00*v10^2/30 + 2*u0*u1*v00*v10/15 - u0*u1*v00/6 + u0*u1*v01^3/45 +
		u0*u1*v01^2*v11/30 - u0*u1*v01^2/10 + u0*u1*v01*v11^2/30 -
		2*u0*u1*v01*v11/15 + u0*u1*v01/6 - u0*u1*v10^3/45 + u0*u1*v10^2/10 -
		u0*u1*v10/6 + u0*u1*v11^3/45 - u0*u1*v11^2/10 + u0*u1*v11/6 -
		u1^2*v00^3/180 - u1^2*v00^2*v10/60 + u1^2*v00^2/30 -
		u1^2*v00*v10^2/30 + u1^2*v00*v10/10 - u1^2*v00/12 + u1^2*v01^3/180 +
		u1^2*v01^2*v11/60 - u1^2*v01^2/30 + u1^2*v01*v11^2/30 -
		u1^2*v01*v11/10 + u1^2*v01/12 - u1^2*v10^3/18 + u1^2*v10^2/5 -
		u1^2*v10/4 + u1^2*v11^3/18 - u1^2*v11^2/5 + u1^2*v11/4

	Q34 = u0^2*v00^3/18 + u0^2*v00^2*v10/30 - u0^2*v00^2/10 +
		u0^2*v00*v10^2/60 - u0^2*v00*v10/20 - u0^2*v01^3/18 -
		u0^2*v01^2*v11/30 + u0^2*v01^2/10 - u0^2*v01*v11^2/60 +
		u0^2*v01*v11/20 + u0^2*v10^3/180 - u0^2*v10^2/60 - u0^2*v11^3/180 +
		u0^2*v11^2/60 + u0*u1*v00^3/45 + u0*u1*v00^2*v10/30 - u0*u1*v00^2/20 +
		u0*u1*v00*v10^2/30 - u0*u1*v00*v10/15 - u0*u1*v01^3/45 -
		u0*u1*v01^2*v11/30 + u0*u1*v01^2/20 - u0*u1*v01*v11^2/30 +
		u0*u1*v01*v11/15 + u0*u1*v10^3/45 - u0*u1*v10^2/20 - u0*u1*v11^3/45 +
		u0*u1*v11^2/20 + u1^2*v00^3/180 + u1^2*v00^2*v10/60 - u1^2*v00^2/60 +
		u1^2*v00*v10^2/30 - u1^2*v00*v10/20 - u1^2*v01^3/180 -
		u1^2*v01^2*v11/60 + u1^2*v01^2/60 - u1^2*v01*v11^2/30 +
		u1^2*v01*v11/20 + u1^2*v10^3/18 - u1^2*v10^2/10 - u1^2*v11^3/18 +
		u1^2*v11^2/10

	Q44 = -u0^2*v00^3/18 - u0^2*v00^2*v10/30 - u0^2*v00*v10^2/60 +
		u0^2*v01^3/18 + u0^2*v01^2*v11/30 + u0^2*v01*v11^2/60 -
		u0^2*v10^3/180 + u0^2*v11^3/180 - u0*u1*v00^3/45 -
		u0*u1*v00^2*v10/30 - u0*u1*v00*v10^2/30 + u0*u1*v01^3/45 +
		u0*u1*v01^2*v11/30 + u0*u1*v01*v11^2/30 - u0*u1*v10^3/45 +
		u0*u1*v11^3/45 - u1^2*v00^3/180 - u1^2*v00^2*v10/60 -
		u1^2*v00*v10^2/30 + u1^2*v01^3/180 + u1^2*v01^2*v11/60 +
		u1^2*v01*v11^2/30 - u1^2*v10^3/18 + u1^2*v11^3/18

	return (u1 - u0) *
				 [ 	Q11 Q12 Q13 Q14 
						Q12 Q22 Q23 Q24 
						Q13 Q23 Q33 Q34 
						Q14 Q24 Q34 Q44	]
end


# finds the ground truth vector for the trapezoid
# NOTE: u0, u1, v00, v01, v10, and v11 are all on [0,1] 
# u0  - the position of the left boundary of the trapezoid 
# u1  - the position of the right boundary of the trapezoid 
# v00 - the height of the bottom left vertex of the trapezoid
# v01 - the height of the top left vertex of the trapezoid 
# v10 - the height of the bottom right vertex of the trapezoid 
# v11 - the height of the top right vertex of the trapezoid 
# p00 - the value at the bottom left vertex of the trapezoid 
# p01 - the value at the top left vertex of the trapezoid 
# p10 - the value at the bottom right vertex of the trapezoid 
# p11 - the value at the top right vertex of the trapezoid
function find_trapezoid_gt_value(u0::Float64, u1::Float64, v00::Float64,
		v01::Float64, v10::Float64, v11::Float64, p00::Float64, p01::Float64,
		p10::Float64, p11::Float64) 
	# this math comes from ogi_gnarly_integrals.py
	T1 = -u0*v00^2*p00/15 - u0*v00^2*p01/30 - u0*v00^2*p10/60 -
		u0*v00^2*p11/120 + u0*v00*v01*p00/30 - u0*v00*v01*p01/30 +
		u0*v00*v01*p10/120 - u0*v00*v01*p11/120 - u0*v00*v10*p00/30 -
		u0*v00*v10*p01/60 - u0*v00*v10*p10/45 - u0*v00*v10*p11/90 +
		u0*v00*v11*p00/120 - u0*v00*v11*p01/120 + u0*v00*v11*p10/180 -
		u0*v00*v11*p11/180 + u0*v00*p00/8 + u0*v00*p01/8 + u0*v00*p10/24 +
		u0*v00*p11/24 + u0*v01^2*p00/30 + u0*v01^2*p01/15 + u0*v01^2*p10/120 +
		u0*v01^2*p11/60 + u0*v01*v10*p00/120 - u0*v01*v10*p01/120 +
		u0*v01*v10*p10/180 - u0*v01*v10*p11/180 + u0*v01*v11*p00/60 +
		u0*v01*v11*p01/30 + u0*v01*v11*p10/90 + u0*v01*v11*p11/45 - u0*v01*p00/8 -
		u0*v01*p01/8 - u0*v01*p10/24 - u0*v01*p11/24 - u0*v10^2*p00/90 -
		u0*v10^2*p01/180 - u0*v10^2*p10/60 - u0*v10^2*p11/120 + u0*v10*v11*p00/180 - 
		u0*v10*v11*p01/180 + u0*v10*v11*p10/120 - u0*v10*v11*p11/120 +
		u0*v10*p00/24 + u0*v10*p01/24 + u0*v10*p10/24 + u0*v10*p11/24 +
		u0*v11^2*p00/180 + u0*v11^2*p01/90 + u0*v11^2*p10/120 + u0*v11^2*p11/60 -
		u0*v11*p00/24 - u0*v11*p01/24 - u0*v11*p10/24 - u0*v11*p11/24 -
		u1*v00^2*p00/60 - u1*v00^2*p01/120 - u1*v00^2*p10/90 - u1*v00^2*p11/180 +
		u1*v00*v01*p00/120 - u1*v00*v01*p01/120 + u1*v00*v01*p10/180 -
		u1*v00*v01*p11/180 - u1*v00*v10*p00/45 - u1*v00*v10*p01/90 -
		u1*v00*v10*p10/30 - u1*v00*v10*p11/60 + u1*v00*v11*p00/180 -
		u1*v00*v11*p01/180 + u1*v00*v11*p10/120 - u1*v00*v11*p11/120 +
		u1*v00*p00/24 + u1*v00*p01/24 + u1*v00*p10/24 + u1*v00*p11/24 +
		u1*v01^2*p00/120 + u1*v01^2*p01/60 + u1*v01^2*p10/180 + u1*v01^2*p11/90 +
		u1*v01*v10*p00/180 - u1*v01*v10*p01/180 + u1*v01*v10*p10/120 -
		u1*v01*v10*p11/120 + u1*v01*v11*p00/90 + u1*v01*v11*p01/45 +
		u1*v01*v11*p10/60 + u1*v01*v11*p11/30 - u1*v01*p00/24 - u1*v01*p01/24 -
		u1*v01*p10/24 - u1*v01*p11/24 - u1*v10^2*p00/60 - u1*v10^2*p01/120 -
		u1*v10^2*p10/15 - u1*v10^2*p11/30 + u1*v10*v11*p00/120 -
		u1*v10*v11*p01/120 + u1*v10*v11*p10/30 - u1*v10*v11*p11/30 + u1*v10*p00/24 +
		u1*v10*p01/24 + u1*v10*p10/8 + u1*v10*p11/8 + u1*v11^2*p00/120 +
		u1*v11^2*p01/60 + u1*v11^2*p10/30 + u1*v11^2*p11/15 - u1*v11*p00/24 -
		u1*v11*p01/24 - u1*v11*p10/8 - u1*v11*p11/8 + v00^2*p00/12 + v00^2*p01/24 +
		v00^2*p10/36 + v00^2*p11/72 - v00*v01*p00/24 + v00*v01*p01/24 -
		v00*v01*p10/72 + v00*v01*p11/72 + v00*v10*p00/18 + v00*v10*p01/36 +
		v00*v10*p10/18 + v00*v10*p11/36 - v00*v11*p00/72 + v00*v11*p01/72 -
		v00*v11*p10/72 + v00*v11*p11/72 - v00*p00/6 - v00*p01/6 - v00*p10/12 -
		v00*p11/12 - v01^2*p00/24 - v01^2*p01/12 - v01^2*p10/72 - v01^2*p11/36 -
		v01*v10*p00/72 + v01*v10*p01/72 - v01*v10*p10/72 + v01*v10*p11/72 -
		v01*v11*p00/36 - v01*v11*p01/18 - v01*v11*p10/36 - v01*v11*p11/18 +
		v01*p00/6 + v01*p01/6 + v01*p10/12 + v01*p11/12 + v10^2*p00/36 +
		v10^2*p01/72 + v10^2*p10/12 + v10^2*p11/24 - v10*v11*p00/72 +
		v10*v11*p01/72 - v10*v11*p10/24 + v10*v11*p11/24 - v10*p00/12 - v10*p01/12 -
		v10*p10/6 - v10*p11/6 - v11^2*p00/72 - v11^2*p01/36 - v11^2*p10/24 -
		v11^2*p11/12 + v11*p00/12 + v11*p01/12 + v11*p10/6 + v11*p11/6
	
	T2 = u0*v00^2*p00/15 + u0*v00^2*p01/30 + u0*v00^2*p10/60 +
		u0*v00^2*p11/120 - u0*v00*v01*p00/30 + u0*v00*v01*p01/30 -
		u0*v00*v01*p10/120 + u0*v00*v01*p11/120 + u0*v00*v10*p00/30 +
		u0*v00*v10*p01/60 + u0*v00*v10*p10/45 + u0*v00*v10*p11/90 -
		u0*v00*v11*p00/120 + u0*v00*v11*p01/120 - u0*v00*v11*p10/180 +
		u0*v00*v11*p11/180 - u0*v01^2*p00/30 - u0*v01^2*p01/15 - u0*v01^2*p10/120 -
		u0*v01^2*p11/60 - u0*v01*v10*p00/120 + u0*v01*v10*p01/120 -
		u0*v01*v10*p10/180 + u0*v01*v10*p11/180 - u0*v01*v11*p00/60 -
		u0*v01*v11*p01/30 - u0*v01*v11*p10/90 - u0*v01*v11*p11/45 +
		u0*v10^2*p00/90 + u0*v10^2*p01/180 + u0*v10^2*p10/60 + u0*v10^2*p11/120 -
		u0*v10*v11*p00/180 + u0*v10*v11*p01/180 - u0*v10*v11*p10/120 +
		u0*v10*v11*p11/120 - u0*v11^2*p00/180 - u0*v11^2*p01/90 - u0*v11^2*p10/120 -
		u0*v11^2*p11/60 + u1*v00^2*p00/60 + u1*v00^2*p01/120 + u1*v00^2*p10/90 +
		u1*v00^2*p11/180 - u1*v00*v01*p00/120 + u1*v00*v01*p01/120 -
		u1*v00*v01*p10/180 + u1*v00*v01*p11/180 + u1*v00*v10*p00/45 +
		u1*v00*v10*p01/90 + u1*v00*v10*p10/30 + u1*v00*v10*p11/60 -
		u1*v00*v11*p00/180 + u1*v00*v11*p01/180 - u1*v00*v11*p10/120 +
		u1*v00*v11*p11/120 - u1*v01^2*p00/120 - u1*v01^2*p01/60 - u1*v01^2*p10/180 -
		u1*v01^2*p11/90 - u1*v01*v10*p00/180 + u1*v01*v10*p01/180 -
		u1*v01*v10*p10/120 + u1*v01*v10*p11/120 - u1*v01*v11*p00/90 -
		u1*v01*v11*p01/45 - u1*v01*v11*p10/60 - u1*v01*v11*p11/30 +
		u1*v10^2*p00/60 + u1*v10^2*p01/120 + u1*v10^2*p10/15 + u1*v10^2*p11/30 -
		u1*v10*v11*p00/120 + u1*v10*v11*p01/120 - u1*v10*v11*p10/30 +
		u1*v10*v11*p11/30 - u1*v11^2*p00/120 - u1*v11^2*p01/60 - u1*v11^2*p10/30 -
		u1*v11^2*p11/15 - v00^2*p00/12 - v00^2*p01/24 - v00^2*p10/36 -
		v00^2*p11/72 + v00*v01*p00/24 - v00*v01*p01/24 + v00*v01*p10/72 -
		v00*v01*p11/72 - v00*v10*p00/18 - v00*v10*p01/36 - v00*v10*p10/18 -
		v00*v10*p11/36 + v00*v11*p00/72 - v00*v11*p01/72 + v00*v11*p10/72 -
		v00*v11*p11/72 + v01^2*p00/24 + v01^2*p01/12 + v01^2*p10/72 + v01^2*p11/36 +
		v01*v10*p00/72 - v01*v10*p01/72 + v01*v10*p10/72 - v01*v10*p11/72 +
		v01*v11*p00/36 + v01*v11*p01/18 + v01*v11*p10/36 + v01*v11*p11/18 -
		v10^2*p00/36 - v10^2*p01/72 - v10^2*p10/12 - v10^2*p11/24 + v10*v11*p00/72 -
		v10*v11*p01/72 + v10*v11*p10/24 - v10*v11*p11/24 + v11^2*p00/72 +
		v11^2*p01/36 + v11^2*p10/24 + v11^2*p11/12 
	
	T3 = u0*v00^2*p00/15 + u0*v00^2*p01/30 + u0*v00^2*p10/60 +
		u0*v00^2*p11/120 - u0*v00*v01*p00/30 + u0*v00*v01*p01/30 -
		u0*v00*v01*p10/120 + u0*v00*v01*p11/120 + u0*v00*v10*p00/30 +
		u0*v00*v10*p01/60 + u0*v00*v10*p10/45 + u0*v00*v10*p11/90 -
		u0*v00*v11*p00/120 + u0*v00*v11*p01/120 - u0*v00*v11*p10/180 +
		u0*v00*v11*p11/180 - u0*v00*p00/8 - u0*v00*p01/8 - u0*v00*p10/24 -
		u0*v00*p11/24 - u0*v01^2*p00/30 - u0*v01^2*p01/15 - u0*v01^2*p10/120 -
		u0*v01^2*p11/60 - u0*v01*v10*p00/120 + u0*v01*v10*p01/120 -
		u0*v01*v10*p10/180 + u0*v01*v10*p11/180 - u0*v01*v11*p00/60 -
		u0*v01*v11*p01/30 - u0*v01*v11*p10/90 - u0*v01*v11*p11/45 + u0*v01*p00/8 +
		u0*v01*p01/8 + u0*v01*p10/24 + u0*v01*p11/24 + u0*v10^2*p00/90 +
		u0*v10^2*p01/180 + u0*v10^2*p10/60 + u0*v10^2*p11/120 - u0*v10*v11*p00/180 + 
		u0*v10*v11*p01/180 - u0*v10*v11*p10/120 + u0*v10*v11*p11/120 -
		u0*v10*p00/24 - u0*v10*p01/24 - u0*v10*p10/24 - u0*v10*p11/24 -
		u0*v11^2*p00/180 - u0*v11^2*p01/90 - u0*v11^2*p10/120 - u0*v11^2*p11/60 +
		u0*v11*p00/24 + u0*v11*p01/24 + u0*v11*p10/24 + u0*v11*p11/24 +
		u1*v00^2*p00/60 + u1*v00^2*p01/120 + u1*v00^2*p10/90 + u1*v00^2*p11/180 -
		u1*v00*v01*p00/120 + u1*v00*v01*p01/120 - u1*v00*v01*p10/180 +
		u1*v00*v01*p11/180 + u1*v00*v10*p00/45 + u1*v00*v10*p01/90 +
		u1*v00*v10*p10/30 + u1*v00*v10*p11/60 - u1*v00*v11*p00/180 +
		u1*v00*v11*p01/180 - u1*v00*v11*p10/120 + u1*v00*v11*p11/120 -
		u1*v00*p00/24 - u1*v00*p01/24 - u1*v00*p10/24 - u1*v00*p11/24 -
		u1*v01^2*p00/120 - u1*v01^2*p01/60 - u1*v01^2*p10/180 - u1*v01^2*p11/90 -
		u1*v01*v10*p00/180 + u1*v01*v10*p01/180 - u1*v01*v10*p10/120 +
		u1*v01*v10*p11/120 - u1*v01*v11*p00/90 - u1*v01*v11*p01/45 -
		u1*v01*v11*p10/60 - u1*v01*v11*p11/30 + u1*v01*p00/24 + u1*v01*p01/24 +
		u1*v01*p10/24 + u1*v01*p11/24 + u1*v10^2*p00/60 + u1*v10^2*p01/120 +
		u1*v10^2*p10/15 + u1*v10^2*p11/30 - u1*v10*v11*p00/120 +
		u1*v10*v11*p01/120 - u1*v10*v11*p10/30 + u1*v10*v11*p11/30 - u1*v10*p00/24 - 
		u1*v10*p01/24 - u1*v10*p10/8 - u1*v10*p11/8 - u1*v11^2*p00/120 -
		u1*v11^2*p01/60 - u1*v11^2*p10/30 - u1*v11^2*p11/15 + u1*v11*p00/24 +
		u1*v11*p01/24 + u1*v11*p10/8 + u1*v11*p11/8
	
	T4 = -u0*v00^2*p00/15 - u0*v00^2*p01/30 - u0*v00^2*p10/60 -
		u0*v00^2*p11/120 + u0*v00*v01*p00/30 - u0*v00*v01*p01/30 +
		u0*v00*v01*p10/120 - u0*v00*v01*p11/120 - u0*v00*v10*p00/30 -
		u0*v00*v10*p01/60 - u0*v00*v10*p10/45 - u0*v00*v10*p11/90 +
		u0*v00*v11*p00/120 - u0*v00*v11*p01/120 + u0*v00*v11*p10/180 -
		u0*v00*v11*p11/180 + u0*v01^2*p00/30 + u0*v01^2*p01/15 + u0*v01^2*p10/120 + 
		u0*v01^2*p11/60 + u0*v01*v10*p00/120 - u0*v01*v10*p01/120 +
		u0*v01*v10*p10/180 - u0*v01*v10*p11/180 + u0*v01*v11*p00/60 +
		u0*v01*v11*p01/30 + u0*v01*v11*p10/90 + u0*v01*v11*p11/45 -
		u0*v10^2*p00/90 - u0*v10^2*p01/180 - u0*v10^2*p10/60 - u0*v10^2*p11/120 +
		u0*v10*v11*p00/180 - u0*v10*v11*p01/180 + u0*v10*v11*p10/120 -
		u0*v10*v11*p11/120 + u0*v11^2*p00/180 + u0*v11^2*p01/90 + u0*v11^2*p10/120 + 
		u0*v11^2*p11/60 - u1*v00^2*p00/60 - u1*v00^2*p01/120 - u1*v00^2*p10/90 -
		u1*v00^2*p11/180 + u1*v00*v01*p00/120 - u1*v00*v01*p01/120 +
		u1*v00*v01*p10/180 - u1*v00*v01*p11/180 - u1*v00*v10*p00/45 -
		u1*v00*v10*p01/90 - u1*v00*v10*p10/30 - u1*v00*v10*p11/60 +
		u1*v00*v11*p00/180 - u1*v00*v11*p01/180 + u1*v00*v11*p10/120 -
		u1*v00*v11*p11/120 + u1*v01^2*p00/120 + u1*v01^2*p01/60 + u1*v01^2*p10/180 + 
		u1*v01^2*p11/90 + u1*v01*v10*p00/180 - u1*v01*v10*p01/180 +
		u1*v01*v10*p10/120 - u1*v01*v10*p11/120 + u1*v01*v11*p00/90 +
		u1*v01*v11*p01/45 + u1*v01*v11*p10/60 + u1*v01*v11*p11/30 -
		u1*v10^2*p00/60 - u1*v10^2*p01/120 - u1*v10^2*p10/15 - u1*v10^2*p11/30 +
		u1*v10*v11*p00/120 - u1*v10*v11*p01/120 + u1*v10*v11*p10/30 -
		u1*v10*v11*p11/30 + u1*v11^2*p00/120 + u1*v11^2*p01/60 + u1*v11^2*p10/30 +
		u1*v11^2*p11/15
	
	return (u1 - u0) * 
					[ T1 
						T2 
						T3 
						T4 ] 
end


# take a point in the uv space of ([0,1], [0,1]) and convert it to the
# rectangular space of ([1,width], [1,height]), since Julia uses 
# 1 - based indexing
function scale_uv(dims::Tuple{Int64,Int64}, p::Point2D{Float64})
	return Point2D{Float64}(p.x * (dims[1] - 1) + 1, p.y * (dims[2] - 1) + 1)
end

# Helper function to get TriangleModel faces
# The triangle face is of the form
# ((Point2D, Point3D), (Point2D, Point3D), (Point2D, Point3D))
# with the first point being (u,v) and the second point being (x,y,z)
function get_obj_tri(obj::TriangleModel, i::Int64, dims::Tuple{Int64,Int64})
	# TriangleModel is just an indexed face set
	return 	((scale_uv(dims, obj.texcoords[obj.face_texcoords[i].x]), 
						obj.positions[obj.face_positions[i].x]),
					(scale_uv(dims, obj.texcoords[obj.face_texcoords[i].y]), 
						obj.positions[obj.face_positions[i].y]),
					(scale_uv(dims, obj.texcoords[obj.face_texcoords[i].z]), 
					 obj.positions[obj.face_positions[i].z]))
end

# determines if a point (p) is in the square with the corners s1 and s4
function point_in_square(p::Point2D{Float64}, s1::Point2D{Float64}, 
		s4::Point2D{Float64})
	return s1.x <= p.x && p.x <= s4.x && s1.y <= p.y && p.y <= s4.y
end

# returns as a Point3D the barycentric coordinates of the point
# on the triangle
# p - the point to find the barycentric coordinate of
# t1, t2, t3 - the points defining the triangle
function get_barry(p::Point2D{Float64}, t1::Point2D{Float64},
		t2::Point2D{Float64}, t3::Point2D{Float64})
	# using Cramer's rule to solve the following system
	# | x1 x2 x3 | | λ1	|		| x |
	# | y1 y2 y3 | | λ2	| = | y |
	# | 1  1  1  | | λ3 |		| 1 |
	
	denom = (t2.y - t3.y) * (t1.x - t3.x) + (t3.x - t2.x) * (t1.y - t3.y)
	
	λ1 = ((t2.y - t3.y) * (p.x - t3.x) + (t3.x - t2.x) * (p.y - t3.y)) / denom
	λ2 = ((t3.y - t1.y) * (p.x - t3.x) + (t1.x - t3.x) * (p.y - t3.y)) / denom
	λ3 = 1.0 - λ1 - λ2
	
	# check for a valid solution
	if -EPS > λ1 || λ1 > 1 + EPS || 
		 -EPS > λ2 || λ2 > 1 + EPS || 
		 -EPS > λ3 || λ3 > 1 + EPS ||
		 isnan(λ1) || isnan(λ2) || isnan(λ3)
		
		 return nothing
	end 

	return Point3D{Float64}(λ1, λ2, λ3)
end


# Returns the point of intersection if the line segments intersect.
# Returns nothing if there is no intersection.
# p1, p2 - points defining the first line segment
# p3, p4 - points defining the second line segment
function lines_intersect(p1,p2,p3,p4)
	# https://en.wikipedia.org/wiki/Line-line_intersection
	t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) /
			((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x))
	
	if t < 0.0  || t > 1.0 || isnan(t)
		return nothing
	end

	u = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) /
			((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x))
	
	if u < 0.0 || u > 1.0 || isnan(u)
		return nothing
	end

	return Point2D{Float64}(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
end

# helper function for find_triangle_square_overlap
# if an intersection is found between the lines defined by p1,p2
# and p3,p4, the intersection is added to i10ns
function add_intersection!(i10ns, p1, p2, p3, p4, t1, t2, t3)
	i10n = lines_intersect(p1,p2,p3,p4)
	if i10n != nothing
		push!(i10ns, Point5D{Float64}(i10n, get_barry(i10n, t1, t2, t3)))
	end
end

# Finds the points of overlap between a square and a triangle
# The returned points are sorted by u-value and the barycentric
# coordinates for the points of overlap are provided
# t1, t2, t3 - points defining the triangle
# s1, s4 - points defining the two corners of the square
function find_triangle_square_overlap(t1::Point2D{Float64},
		t2::Point2D{Float64}, t3::Point2D{Float64}, s1::Point2D{Float64},
		s4::Point2D{Float64})
	i10ns = Vector{Point5D{Float64}}()
	s2 = Point2D{Float64}(s1.x, s4.y)
	s3 = Point2D{Float64}(s4.x, s1.y)


	# There are 19 possible points defining the area of overlap:
	# 1 - 12) 	The intersections between the sides of the shapes
	# 13 - 16) 	The four vertices of the square
	# 17 - 19) 	The three vertices of the triangle
	# The maximum number of overlap points at any time is 6
	# The minimum number of overlap points at any time is 3

	# t1, t2 and square left
	add_intersection!(i10ns, t1, t2, s1, s2, t1, t2, t3)
	
	# t1, t2 and square top
	add_intersection!(i10ns, t1, t2, s2, s4, t1, t2, t3)
	
	# t1, t2 and square right
	add_intersection!(i10ns, t1, t2, s3, s4, t1, t2, t3)

	# t1, t2 and square bottom
	add_intersection!(i10ns, t1, t2, s1, s3, t1, t2, t3)
	
	# t1, t3 and square left
	add_intersection!(i10ns, t1, t3, s1, s2, t1, t2, t3) 
	
	# t1, t3 and square top
	add_intersection!(i10ns, t1, t3, s2, s4, t1, t2, t3)
	
	# t1, t3 and square right
	add_intersection!(i10ns, t1, t3, s3, s4, t1, t2, t3)

	# t1, t3 and square bottom
	add_intersection!(i10ns, t1, t3, s1, s3, t1, t2, t3) 
	
	# t2, t3 and square left
	add_intersection!(i10ns, t2, t3, s1, s2, t1, t2, t3)
	
	# t2, t3 and square top
	add_intersection!(i10ns, t2, t3, s2, s4, t1, t2, t3) 
	
	# t2, t3 and square right
	add_intersection!(i10ns, t2, t3, s3, s4, t1, t2, t3) 

	# t2, t3 and square bottom
	add_intersection!(i10ns, t2, t3, s1, s3, t1, t2, t3) 

	if length(i10ns) < 6
		# let's check all the vertices
		if point_in_square(t1,s1,s4)
			push!(i10ns, Point5D{Float64}(t1,1.0,0.0,0.0))
		end
		if point_in_square(t2,s1,s4)
			push!(i10ns, Point5D{Float64}(t2,0.0,1.0,0.0))
		end
		if point_in_square(t3,s1,s4)
			push!(i10ns, Point5D{Float64}(t3,0.0,0.0,1.0))
		end
		
		b1 = get_barry(s1, t1, t2, t3)
		
		if b1 != nothing
			push!(i10ns, Point5D{Float64}(s1, b1))
		end

		s2 = Point2D(s1.x, s4.y)
		b2 = get_barry(s2, t1, t2, t3)
		
		if b2 != nothing
			push!(i10ns, Point5D{Float64}(s2, b2))
		end
		
		s3 = Point2D(s4.x, s1.y)
		b3 = get_barry(s3, t1, t2, t3)
		
		if b3 != nothing
			push!(i10ns, Point5D{Float64}(s3, b3))
		end
		
		b4 = get_barry(s4, t1, t2, t3)
		
		if b4 != nothing
			push!(i10ns, Point5D{Float64}(s4, b4))
		end
	end

	# we don't actually have overlap
	if length(i10ns) < 3
		return nothing
	end
	
	return unique(i10ns)
end

# helper function for create trapezoids
function is_bottom_point(min_x, max_x, i)
	if min_x < i && i < max_x
		return true
	end

	return false
end

function find_t(x1, x2, x)
	# Math derived from linear interpolation: 
	# 	x = (1 - t) * x1 + t * x2
	# 	x = x1 - t * x1 + t * x2
	# 	x - x1 = t(x2 - x1)
	# 	t = (x - x1) / (x2 - x1)
	return (x - x1) / (x2 - x1)
end

# Returns the trapezoidal slices created by a collection of points
# that defined a convex shape
# v - the points of overlap between the square the triangle
# 		ideally returned by find_triangle_square_overlap
function create_trapezoids(v)
	# we don't have enough points to make trapezoids
	if length(v) < 3
		return []
	end
	
	# ensure the points are in the order we want them
	sort!(v, lt=(a,b) -> a.u < b.u ? true : 
				a.u == b.u && a.v < b.v)

	# find the centroid so we can determine the counter-clock-wise order of points 
	centroid = sum((x -> [x.u, x.v]).(v)) / Float64(length(v))
	ccw_indices = sortperm(map(x -> atan(x.v - centroid[2], x.u - centroid[1]), v)) 
	min_x = first(ccw_indices)
	max_x = last(ccw_indices)

	# each trapezoid is a vector of 4 5D Points
	# the u,v coords are the u,v coords
	# the x,y,z coords are the bayesian coords
	trapezoids = Vector{Vector{Point5D{Float64}}}()
	
	# bottom_left, top_left, top_right, and bottom_right
	# are all indicess into the point vector
	bottom_left = v[1]
	top_left = v[1]

	top_right = nothing
	bottom_right = nothing

	i = 2

	# check if the left side is square
	if v[2].u == v[1].u
		i = 3
		top_left = v[2]
	end

	len = length(v)
	rs_square = false	# [r]ight [s]ide square

	# check if the right side is square
	if v[len - 1].u == v[len].u
		len -= 1
		rs_square = true
	end

	# iterate through all the points
	while i < len
		if is_bottom_point(min_x, max_x, ccw_indices[i])
			# need to find top_right
			if top_right == nothing
				j = i + 1
				
				while is_bottom_point(min_x, max_x, ccw_indices[j]) && j < len
					j += 1
				end

				if j == len && rs_square
					top_right = v[j+1]
				else
					top_right = v[j]
				end
			end
			# interpolate the top point
			t = find_t(top_left.u, top_right.u, v[i].u)
			interp = (1 - t) * top_left + t * top_right

			push!(trapezoids, 
						[bottom_left,
						 top_left,
						 v[i],
						 interp])
			
			# update left points and clear bottom_right
			bottom_left = v[i]
			top_left = interp

			bottom_right = nothing
		else
			# need to find bottom_right
			if bottom_right == nothing
				j = i + 1
				while !is_bottom_point(min_x, max_x, ccw_indices[j]) && j < len
					j += 1
				end

				bottom_right = v[j]
			end
			# interpolate bottom point
			t = find_t(bottom_left.u, bottom_right.u, v[i].u)
			interp = (1 - t) * bottom_left + t * bottom_right

			push!(trapezoids, 
						[bottom_left,
						 top_left,
						 interp,
						 v[i]])
			
			# update left points and clear top_right
			bottom_left = interp
			top_left = v[i]

			top_right = nothing
		end
		
		i += 1
	end

	# create the last trapezoid
	if rs_square
		top_right = v[len + 1]
	else
		top_right = v[len]
	end

	bottom_right = v[len]

	push!(trapezoids,
				[bottom_left, top_left, bottom_right, top_right])
	
	return trapezoids
end

# helper function to update an error matrix
# m - the matrix to update
# q - the error matrix from find_trapezoid_bi_value
# ts1,ts2,ts3,ts4 - the indices of the texel squares involved
function update_errmat!(m, q, ts1::Int, ts2::Int, ts3::Int, ts4::Int) 
	# NOTE: Access in col-major order for performance

	m[ts1,ts1] += q[1,1]
	m[ts2,ts1] += q[2,1]
	m[ts3,ts1] += q[3,1] 
	m[ts4,ts1] += q[4,1]

	m[ts1,ts2] += q[1,2]
	m[ts2,ts2] += q[2,2]
	m[ts3,ts2] += q[3,2]
	m[ts4,ts2] += q[4,2]

	m[ts1,ts3] += q[1,3]
	m[ts2,ts3] += q[2,3]
	m[ts3,ts3] += q[3,3]
	m[ts4,ts3] += q[4,3]

	m[ts1,ts4] += q[1,4]
	m[ts2,ts4] += q[2,4]
	m[ts3,ts4] += q[3,4]
	m[ts4,ts4] += q[4,4]
end

# helper function to update a truth vector
# v - the vector to update
# t - the truth values from find_trapezoid_gt_value
# ts1,ts2,ts3,ts4 - the indices of the texel squares involved
function update_truthvec!(v, t, ts1::Int, ts2::Int, ts3::Int, ts4::Int)
	v[ts1] += t[1]
	v[ts2] += t[2]
	v[ts3] += t[3]
	v[ts4] += t[4]
end

# modifies the error matrix and truth vectors with the error and truth
# values for the given triangle
# mat - the x,y,z error matrix to update
# vecs - the x,y,z truth vectors to update
# triangle - a triangle of the form given by get_obj_tri
# dims - the dimensions of the final geometry image
function find_triangle_error!(mat, vecs, triangle, dims)
	# steps:
	# 1) Find all the texel squares that intersect with the triangle
	# 2) For each overlapping square, break the overlapping area into
	# 	 trapezoidal slices
	# #	2.1) Find the x,y,z coordinates of the trapezoid corners
	# 3) For each slice, find the error matrix and truth vectors accordingly
	# 4) Update errmats and truthvecs

	# find the bounding box for the triangle
	umin = floor(min(triangle[1][1].x, triangle[2][1].x, triangle[3][1].x))
	umax = floor(max(triangle[1][1].x, triangle[2][1].x, triangle[3][1].x))
	vmin = floor(min(triangle[1][1].y, triangle[2][1].y, triangle[3][1].y))
	vmax = floor(max(triangle[1][1].y, triangle[2][1].y, triangle[3][1].y))	

	# iterate through the bounding box
	# NOTE: we are interested in the squares between the texels
	for u in umin:umax-1
		for v in vmin:vmax-1
			overlap = find_triangle_square_overlap(triangle[1][1], triangle[2][1],
																						 triangle[3][1], Point2D(u,v),
																						 Point2D(u+1,v+1))
			if overlap != nothing
				# ts = texel square
				ts1 = Int(dims[2] - v + 1 + (u - 1) * dims[1])
				ts2 = Int(dims[2] - v + 1 + u * dims[1])
				ts3 = Int(dims[2] - v + (u - 1) * dims[1])
				ts4 = Int(dims[2] - v + u * dims[1])
			
				trapezoids = create_trapezoids(overlap)
				
				for t in trapezoids
					# use the bayesian coordinates to create the four xyz points
					p00 = triangle[1][2] * t[1].x + 
								triangle[2][2] * t[1].y + 
								triangle[3][2] * t[1].z

					p01 = triangle[1][2] * t[2].x + 
								triangle[2][2] * t[2].y + 
								triangle[3][2] * t[2].z

					p10 = triangle[1][2] * t[3].x + 
								triangle[2][2] * t[3].y + 
								triangle[3][2] * t[3].z

					p11 = triangle[1][2] * t[4].x + 
								triangle[2][2] * t[4].y + 
								triangle[3][2] * t[4].z

					u0 = t[1].u - u
					u1 = t[4].u - u
					v00 = t[1].v - v
					v01 = t[2].v - v
					v10 = t[3].v - v
					v11 = t[4].v - v
					
					q = find_trapezoid_bi_value(u0, u1, v00, v01, v10, v11)
					update_errmat!(mat, q, ts1, ts2, ts3, ts4)
					
					xt = find_trapezoid_gt_value(u0, u1, v00, v01, v10, v11,
																			 p00.x, p01.x, p10.x, p11.x)
					update_truthvec!(vecs[1], xt, ts1, ts2, ts3, ts4)

					yt = find_trapezoid_gt_value(u0, u1, v00, v01, v10, v11,
																			 p00.y, p01.y, p10.y, p11.y)
					update_truthvec!(vecs[2], yt, ts1, ts2, ts3, ts4)
					
					zt = find_trapezoid_gt_value(u0, u1, v00, v01, v10, v11,
																			 p00.z, p01.z, p10.z, p11.z)
					update_truthvec!(vecs[3], zt, ts1, ts2, ts3, ts4)
				end
			end
		end
	end
end

function get_nz_rows(mat)
	return sort!(unique(rowvals(mat)))
end

# does the impossible: solves a singular matrix
function solve_system(mat, vecs)
	nz_rows = get_nz_rows(mat)

	# sub_mat will be the original matrix with the 
	# rows and cols that are all just 0s removed
	sub_mat_size = size(nz_rows)[1]	
	sub_mat = spzeros(sub_mat_size, sub_mat_size)
	sub_vecs = (zeros(sub_mat_size),
							zeros(sub_mat_size),
							zeros(sub_mat_size))

	sub_j = 1

	vals = nonzeros(mat)
	rows = rowvals(mat)

	n = size(mat)[1]
	
	# fill in sub_mat
	for j = 1:n
		r = nzrange(mat, j)
		if r != 1:0
			for i in r
				sub_i = findfirst(x -> x == rows[i], nz_rows)
				sub_mat[sub_i,sub_j] = vals[i]
			end
			
			sub_j += 1
		end
	end
	
	sub_i = 1
	
	# fill in sub_vecs
	for i in nz_rows 
		sub_vecs[1][sub_i] = vecs[1][i] 
		sub_vecs[2][sub_i] = vecs[2][i] 
		sub_vecs[3][sub_i] = vecs[3][i] 

		sub_i += 1
	end
	
	# solve the sub system
	sub_solution = (sub_mat\sub_vecs[1],
									sub_mat\sub_vecs[2],
									sub_mat\sub_vecs[3])
	
	# fill the solution with NaN, so values that we
	# couldn't find will be NaN in the final vector
	solution = (fill(NaN, size(vecs[1])[1]),
							fill(NaN, size(vecs[2])[1]),
							fill(NaN, size(vecs[3])[1]))
	
	# fill in the full solution matrix
	sub_i = 1
	for i in nz_rows
		solution[1][i] = sub_solution[1][sub_i]
		solution[2][i] = sub_solution[2][sub_i]
		solution[3][i] = sub_solution[3][sub_i]
		sub_i += 1
	end

	return solution
end

function nan_max(a)
	if isnan(a)
		return -Inf
	else
		return a
	end
end

function nan_min(a)
	if isnan(a)
		return Inf
	else
		return a
	end
end


# creates a geometry image with the given texels and dimensions
function create_gi(texels, width::Int, height::Int)
	# shift the values so that the min value is 0
	min_x = minimum(nan_min, texels[1])
	min_y = minimum(nan_min, texels[2])
	min_z = minimum(nan_min, texels[3])
	
	# scale by the max of all x,y,z vals
	scaling = max(maximum(nan_max, texels[1]) - min_x,
								maximum(nan_max, texels[2]) - min_y,
								maximum(nan_max, texels[3]) - min_z)

	f = (x,y,z) -> RGB{Float64}(isnan(x) ? 0.0 : ((x - min_x) / scaling),
															isnan(y) ? 0.0 : ((y - min_y) / scaling),
															isnan(z) ? 0.0 : ((z - min_z) / scaling))
	
	# now actually create the image
	image = map(f, texels[1], texels[2], texels[3])
	return reshape(image, (width, height))
end

# converts a TriangleModel to a geometry image with the given dimensions
function convert_obj_to_gi(obj::TriangleModel, width::Int, height::Int)
	n_texels = width * height
	dims = (width, height)
	
	# mats is one n_texel x n_texel sparse matrix
	mat = spzeros(n_texels, n_texels)
	
	# vecs is three vectors of length n_texels)
	vecs = map(x -> spzeros(x), repeat([n_texels],3))

	for i in 1:length(obj.face_positions)
		find_triangle_error!(mat, vecs, get_obj_tri(obj, i, dims), dims)
	end
	
	image = create_gi(solve_system(mat, vecs), width, height)
	save("test-"*string(now())*".png", image)
end
