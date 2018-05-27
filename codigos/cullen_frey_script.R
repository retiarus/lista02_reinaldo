library(rio)
library(fitdistrplus)
library(optparse)

args = commandArgs(trailingOnly=TRUE)

name_txt <- paste(args[1], ".txt", sep = "")
name_png <- paste(args[1], ".png", sep = "")

setwd("/home/peregrinus/Arquivos/ipython/matcomp")
vec <- import(file = name_txt)
png(name_png)
descdist(vec$V1, boot = 1000)
