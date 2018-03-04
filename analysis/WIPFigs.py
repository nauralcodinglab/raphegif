"""
Run ABF test first.
"""

#%% SHOW POOR FIT ON MICHAEL'S DATA

"""
Run ABF test first.
"""

# Get V_traces
Vreal_trace = myExp.testset_traces[0].V
Ireal_trace = myExp.testset_traces[0].I
t_vec, Vsim_trace, _, _, _ = myGIF.simulate(Ireal_trace, myGIF.El)
del _


# Get real spike times.
test_spk_times = []

for tr in myExp.testset_traces:
    
    test_spk_times.append(tr.getSpikeTimes())
    

# Simulate bunch of spikes.
sim_spk_times = []

for i in range(len(myExp.testset_traces)):
    
    sim_spk_times.append(myGIF.simulateSpikingResponse(Ireal_trace, 0.1))


plt.figure(figsize = (8, 5))
traces_I = plt.subplot2grid((5, 1), (0, 0), frameon = False)
traces_V = plt.subplot2grid((5, 1), (1, 0), rowspan = 2, sharex = traces_I, frameon = False)
real = plt.subplot2grid((5, 1), (3, 0), sharex = traces_I, frameon = False)
sim = plt.subplot2grid((5, 1), (4, 0), sharex = traces_I, frameon = False)

traces_I.plot(t_vec, Ireal_trace, 'k-')
traces_V.plot(t_vec, Vreal_trace, 'k-', label = 'Real')
traces_V.plot(t_vec, Vsim_trace, 'r-', label = 'Simulated')
traces_V.legend()

for i in range(len(test_spk_times)):
    
    real.plot(test_spk_times[i], [i] * len(test_spk_times[i]), 'k|')
    sim.plot(sim_spk_times[i], [i] * len(sim_spk_times[i]), 'r|')
    
traces_I.get_xaxis().set_visible(False)
traces_V.get_xaxis().set_visible(False)
real.get_xaxis().set_visible(False)    

plt.tight_layout()
plt.show()

#%% SHOW SPIKELESS RECORDINGS

# Get V_traces
Vreal_trace = myExp.testset_traces[0].V
Ireal_trace = myExp.testset_traces[0].I
t_vec, Vsim_trace = myGIF.simulate(Ireal_trace, -60)


plt.figure(figsize = (8, 5))
traces_I = plt.subplot2grid((2, 1), (0, 0), frameon = False)
traces_V = plt.subplot2grid((2, 1), (1, 0), sharex = traces_I, frameon = False)

traces_I.plot(t_vec, Ireal_trace, 'k-')

traces_V.plot(t_vec, Vreal_trace, 'k-', label = 'Real')
traces_V.plot(t_vec, Vsim_trace, 'r-', label = 'Simulated')
traces_V.legend()

traces_I.get_xaxis().set_visible(False)

plt.tight_layout()
plt.show()