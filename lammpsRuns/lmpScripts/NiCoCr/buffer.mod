#--- define groups
variable xxlo	equal xlo
variable x0		equal ${xxlo}+${buff}
#
variable xxhi	equal xhi 
variable x1		equal ${xxhi}-${buff}
#
variable yylo	equal ylo
variable y0	equal ${yylo}+${buff}

variable yyhi	equal yhi
variable y1		equal ${yyhi}-${buff}
#
region up block INF INF ${y1} INF INF INF
region down block INF INF INF ${y0} INF INF
#region right block ${x1} INF INF INF INF INF
#region left block INF ${x0} INF INF INF INF
group upp region up
group downn region down
#group lg region left
#group rg region right
#
#--- fix walls
fix 1 upp setforce 0.0 0.0 0.0
fix 2 downn setforce 0.0 0.0 0.0
#fix 11 lg setforce 0.0 0.0 0.0
#fix 22 rg setforce 0.0 0.0 0.0
velocity upp set 0 0 0
velocity downn set 0 0 0
#velocity lg set 0 0 0
#velocity rg set 0 0 0


