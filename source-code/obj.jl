import Base.+
import Base.-
import Base.*
import Base./
import Base.atan

struct Point2D{T<:Real} 
	x::T
	y::T
end

Point2D{T}() where {T<:Real} = Point2D{T}(0,0)
Point2D{T}(v::Vector{T}) where {T<:Real} = Point2D{T}(v[1], v[2])

dot(a::Point2D{T}, b::Point2D{T}) where {T<:Real} = a.x * b.x + a.y * b.y
atan(a::Point2D{T}) where {T<:Real} = atan(a.y, a.x)
+(a::Point2D{T}, b::Point2D{T}) where {T<:Real} = Point2D{T}(a.x + b.x, a.y + b.y)
-(a::Point2D{T}, b::Point2D{T}) where {T<:Real} = Point2D{T}(a.x - b.x, a.y - b.y)
*(a::T, b::Point2D{T}) where {T<:Real} = Point2D{T}(a * b.x, a * b.y)
*(a::Point2D{T}, b::T) where {T<:Real} = Point2D{T}(a.x * b, a.y * b)
/(a::Point2D{T}, b::T) where {T<:Real} = Point2D{T}(a.x / b, a.y / b)

struct Point3D{T<:Real}
	x::T
	y::T
	z::T
end

Point3D{T}() where {T<:Real} = Point3D{T}(0,0,0)
Point3D{T}(v::Vector{T}) where {T<:Real} = Point3D{T}(v[1], v[2], v[3])

dot(a::Point3D{T}, b::Point3D{T}) where {T<:Real} = a.x * b.x + a.y * b.y + a.z * b.z
+(a::Point3D{T}, b::Point3D{T}) where {T<:Real} = Point3D{T}(a.x + b.x, a.y + b.y, a.z + b.z)
-(a::Point3D{T}, b::Point3D{T}) where {T<:Real} = Point3D{T}(a.x - b.x, a.y - b.y, a.z - b.z)
*(a::T, b::Point3D{T}) where {T<:Real} = Point3D{T}(a * b.x, a * b.y, a * b.z)
*(a::Point3D{T}, b::T) where {T<:Real} = Point3D{T}(a.x * b, a.y * b, a.z * b)
/(a::Point3D{T}, b::T) where {T<:Real} = Point3D{T}(a.x / b, a.y / b, a.z / b)

struct Point5D{T<:Real}
	u::T
	v::T
	x::T
	y::T
	z::T
end

Point5D{T}() where {T<:Real} = Point5D{T}(0,0,0,0,0)
Point5D{T}(v::Vector{T}) where {T<:Real} = Point5D{T}(v[1], v[2], v[3], v[4], v[5])
Point5D{T}(a::Point2D{T}, b::Point3D{T}) where {T<:Real} = Point5D{T}(a.x, a.y, b.x, b.y, b.z) 
Point5D{T}(a::Point2D{T}, b::T, c::T, d::T) where {T<:Real} = Point5D{T}(a.x, a.y, b, c, d)

dot(a::Point5D{T}, b::Point5D{T}) where {T<:Real} = a.u * b.u + a.v * b.v + a.x * b.x + a.y * b.y + a.z * b.z
+(a::Point5D{T}, b::Point5D{T}) where {T<:Real} = Point5D{T}(a.u + b.u, a.v + b.v, a.x + b.x, a.y + b.y, a.z + b.z)
-(a::Point5D{T}, b::Point5D{T}) where {T<:Real} = Point5D{T}(a.u - b.u, a.v - b.v, a.x - b.x, a.y - b.y, a.z - b.z)
*(a::T, b::Point5D{T}) where {T<:Real} = Point5D{T}(a * b.u, a * b.v, a * b.x, a * b.y, a * b.z)
*(a::Point5D{T}, b::T) where {T<:Real} = Point5D{T}(a.u * b, a.v * b, a.x * b, a.y * b, a.z * b)
/(a::Point5D{T}, b::T) where {T<:Real} = Point5D{T}(a.u / b, a.v / b, a.x / b, a.y / b, a.z / b)

struct TriangleModel
	positions::Vector{Point3D{Float64}}
	normals::Vector{Point3D{Float64}}
	texcoords::Vector{Point2D{Float64}}
	
	face_positions::Vector{Point3D{Int64}}
	face_normals::Vector{Point3D{Int64}}
	face_texcoords::Vector{Point3D{Int64}}
end

# loads in an object file and returns a TriangleModel
# this simple implementation only supports the v, vt, vn, and f tags
function load_obj_file(filename::AbstractString)
	p = Vector{Point3D{Float64}}()	# positions
	n = Vector{Point3D{Float64}}()	# normals
	t = Vector{Point2D{Float64}}()	# texcoords
	
	fp = Vector{Point3D{Int64}}() # face positions
	fn = Vector{Point3D{Int64}}() # face normals
	ft = Vector{Point3D{Int64}}() # face texcoords
	
	# Wavefront .obj format line prefixes
	# See https://en.wikipedia.org/wiki/Wavefront_.obj_file for more info
	# 	v  -> position, followed by 3 floating point numbers
	# 	vt -> texcoord, followed by 2 floating point numbers
	# 	vn -> normal, followed by 3 floating point numbers
	# 	f  -> face, followed by 3 sets of 3 integers of the form 1/2/3
	v_regex = r"([v][nt]?)[\W]*([0-9]*\.?[0-9]*)[\W]*([0-9]*\.?[0-9]*)[\W]*([0-9]*\.?[0-9]*)?" 
	f_regex = r"f[\W]*([0-9]*)/([0-9]*)/([0-9]*)[\W]*([0-9]*)/([0-9]*)/([0-9]*)[\W]*([0-9]*)/([0-9]*)/([0-9]*)"

	for line in eachline(filename)

		# check for vertex data lines 
		m = match(v_regex, line)

		if m != nothing
			if m[1] == "v"
				push!(p, Point3D{Float64}(parse.(Float64, [m[2], m[3], m[4]])))
			elseif m[1] == "vn"
				push!(n, Point3D{Float64}(parse.(Float64, [m[2], m[3], m[4]])))
			elseif m[1] == "vt"
				push!(t, Point2D{Float64}(parse.(Float64, [m[2], m[3]])))
			end
		else
			m = match(f_regex, line)

			if m != nothing
				pos = parse.(Int64, [m[1], m[4], m[7]])
				tex = parse.(Int64, [m[2], m[5], m[8]])
				norm = parse.(Int64, [m[3], m[6], m[9]])
				
				# check for valid bounds
				if !checkbounds(Bool, p, pos[1]) || 
					!checkbounds(Bool, p, pos[2])  ||
					!checkbounds(Bool, p, pos[3])  ||
					!checkbounds(Bool, t, tex[1])  ||
					!checkbounds(Bool, t, tex[2])  ||
					!checkbounds(Bool, t, tex[3])  ||
					!checkbounds(Bool, n, norm[1]) ||
					!checkbounds(Bool, n, norm[2]) ||
					!checkbounds(Bool, n, norm[3]) 
					throw(error("load_obj_file: Invalid index specified in faces!")) 
				end
			
				push!(fp, Point3D{Int64}(pos))
				push!(ft, Point3D{Int64}(tex))
				push!(fn, Point3D{Int64}(norm))
			else
				# comment or blank line
				if match(r"^(#.*)$|^([\W]*)$", line) != nothing
					continue
				# otherwise it's an invalid line and we should tell someone
				else
					throw(error("load_obj_file: Unexpected line in .obj file!"))	
				end
			end
		end
	end

	return TriangleModel(p, n, t, fp, fn, ft)
end

