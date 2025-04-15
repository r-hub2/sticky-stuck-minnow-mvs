# This file is part of mvs: Methods for High-Dimensional Multi-View Learning
# Copyright (C) 2018-2024  Wouter van Loon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# This is a simple function that translates the MVS adaptive argument to the appropriate STaPLR argument.
translate_adaptive_argument <- function(x){
  if(x==TRUE){
    return("adaptive")
  }else if(x==FALSE){
    return(NULL)
  }
  else{
    stop("Adaptive argument must be of class logical.")
  }
}