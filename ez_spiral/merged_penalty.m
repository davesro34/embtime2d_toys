function out=merged_penalty(tau,real,calc)
out=sum(1-exp(-((real+tau-calc)/5.0).^2));