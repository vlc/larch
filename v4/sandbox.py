

if 1:
	import larch
	import numpy

	# m = larch.Model.Example(22)
	# m.maximize_loglike()

	# d_u_ca = m.data.UtilityCA.copy()
	# d_u_co = m.data.UtilityCO.copy()
	#
	# coef_u_ca = m.Coef("UtilityCA").copy()
	# coef_u_co = m.Coef("UtilityCO").copy()
	#
	#

	dt = larch.DT.Example()

	from larch4.parameter_collection import ParameterCollection
	from larch4.data_collection import DataCollection
	from larch4.workspace_collection import WorkspaceCollection
	from larch4.roles import P,X,PX

	from larch4.model import Model

	altcodes = [1,2,3,4,5,6]
	alterns = dt.alternatives()

	param_names = ['costbyincome',
	               'motorized_time',
	               'nonmotorized_time',
	               'motorized_ovtbydist',
	               'vehbywrk_SR',
	               'mu_motor',
	               'mu_nonmotor',
	               'wkcbd_SR2',
	               'wkempden_SR2',
	               'ASC_SR2',
	               'wkcbd_SR3+',
	               'wkempden_SR3+',
	               'ASC_SR3+',
	               'hhinc#4',
	               'vehbywrk_Tran',
	               'wkcbd_Tran',
	               'wkempden_Tran',
	               'ASC_Tran',
	               'hhinc#5',
	               'vehbywrk_Bike',
	               'wkcbd_Bike',
	               'wkempden_Bike',
	               'ASC_Bike',
	               'hhinc#6',
	               'vehbywrk_Walk',
	               'wkcbd_Walk',
	               'wkempden_Walk',
	               'ASC_Walk']

	param_values = (-0.03861527921549844,
					 -0.014523038624767998,
					 -0.046215369431426574,
					 -0.11379697567750172,
					 -0.2256677903743352,
					 0.7257824243960934,
					 0.768934053868761,
					 0.19313965705114092,
					 0.0011489703923458934,
					 -1.3250178603913265,
					 0.7809570669175268,
					 0.0016377717591202284,
					 -2.5055361149071476,
					 -0.003931169283310062,
					 -0.7070553074051729,
					 0.9213037533865539,
					 0.002236618465113466,
					 -0.4035506229859652,
					 -0.010045677860597766,
					 -0.7347879705726603,
					 0.40770600782114,
					 0.0016743955255113922,
					 -1.2012147989034374,
					 -0.0062076142199297265,
					 -0.7638904490769256,
					 0.11414028210681003,
					 0.0021704169426275174,
					 0.34548388394786805)



	p = Model( datasource=dt, parameters=param_names )



	p.utility_ca = (
		+ P("costbyincome")*X('totcost/hhinc')
		+ X("tottime * (altnum in (1,2,3,4))")*P("motorized_time")
		+ X("tottime * (altnum in (5,6))")*P("nonmotorized_time")
		+ X("ovtt/dist * (altnum in (1,2,3,4))")*P("motorized_ovtbydist")
	)

	p.utility_co[4] += P("hhinc#4") * X("hhinc")
	p.utility_co[5] += P("hhinc#5") * X("hhinc")
	p.utility_co[6] += P("hhinc#6") * X("hhinc")

	for a, name in alterns[1:3]:
		p.utility_co[a] += X("vehbywrk") * P("vehbywrk_SR")
		p.utility_co[a] += X("wkccbd+wknccbd") * P("wkcbd_" + name)
		p.utility_co[a] += X("wkempden") * P("wkempden_" + name)
		p.utility_co[a] += P("ASC_" + name)

	for a, name in alterns[3:]:
		p.utility_co[a] += X("vehbywrk") * P("vehbywrk_" + name)
		p.utility_co[a] += X("wkccbd+wknccbd") * P("wkcbd_" + name)
		p.utility_co[a] += X("wkempden") * P("wkempden_" + name)
		p.utility_co[a] += P("ASC_" + name)


	for p_name, p_value in zip(param_names, param_values):
		p.set_value(p_name, p_value)

	p.set_value('vehbywrk_SR3+', p.get_value('vehbywrk_SR'))
	p.set_value('vehbywrk_SR2', p.get_value('vehbywrk_SR'))

	p.unmangle()


	print([i for i in p._u_ca_varindex])

	# d = DataCollection(
	# 	dt.caseids().copy(), m.df.alternative_codes(),
	# 	p.utility_ca_vars, #  m.needs()['UtilityCA'].get_variables(),
	# 	p.utility_co_vars, # m.needs()['UtilityCO'].get_variables(),
	# )
	#
	# d.load_data(m.df)
	p.load_data()



	from larch4.nesting.tree import NestingTree

	t = NestingTree()
	t.add_nodes([1,2,3,4,5,6])

	t.add_node(7, children=(1,2,3,4), parameter='mu_motor')
	t.add_node(8, children=(5,6), parameter='mu_nonmotor')

	p.graph = t


	# work = WorkspaceCollection(d,p,t)
	# work = p.work




	# p.data._calculate_exp_utility_elemental(p, work.exp_util_elementals)
	# from larch4.nesting.nl_utility import exp_util_of_nests
	# exp_util_of_nests(work.exp_util_elementals,work.exp_util_nests,t,p)

	# p.calculate_exp_utility()
	#
	#
	# from larch4.nesting.nl_prob import conditional_logprob_from_tree, elemental_logprob_from_conditional_logprob
	#
	#
	# conditional_logprob_from_tree(
	# 	work.exp_util_elementals,
	# 	work.exp_util_nests,
	# 		t,
	# 		p,
	# 		work.log_conditional_prob
	# 	)
	# print("?"*30, 'Prob?')
	# # print(m.work.probability[0])
	# print(work.log_conditional_prob[0][0])
	#
	# try:
	# 	print(work.log_conditional_prob[7][0])
	# 	print(work.log_conditional_prob[8][0])
	# except KeyError:
	# 	pass
	#
	# print("!"*30, 'Prob!')
	#
	# elemental_logprob_from_conditional_logprob(
	# 	work.log_conditional_prob,
	# 	t,
	# 	work.log_prob
	#
	# )
	#


	print("-----LL-----")
	LL = p.loglike()
	print(LL)
	assert( numpy.isclose(LL, -3441.6725270750367, rtol=1e-09, atol=1e-09, equal_nan=False) )


	# print(m.loglike())

	# -3441.6725270750367
	# -3441.6725270750376


	print('done')

	# %timeit exp_util_of_nests_threaded(u,u1,t,p,7)
	# %timeit exp_util_of_nests(u,u1,t,p)
	# %prun exp_util_of_nests(u,u1,t,p)


	# %timeit -n 2000 m.loglike(cached=False)
	# %timeit -n 2000 p.loglike()

	# %prun m.loglike(cached=False)
	# %prun [p.loglike() for _ in range(1000)]
	# %load_ext line_profiler
	# %lprun -f go go()


	def go():
		p.loglike()




	# %timeit d._calculate_log_like(work.log_prob)


	print( p.coef_utility_co )
	# print( m.Coef("UtilityCO").squeeze() )

	m = larch.Model.Example(22)
	m.provision()
	#%timeit m.loglike(cached=False)


	#
	#
	# from larch.roles import P,X
	# from larch4.parameter_collection import ParameterCollection
	#
	# names = ['aa',]
	#
	# uca = P.aa * X.aaa + P.bb * X.bbb
	#
	# uco = {
	# 	1: P.co1 * X.coo1,
	# 	2: P.co2 * X.coo2,
	# }
	#
	# p = ParameterCollection(names, utility_ca=uca, utility_co=uco, altindex=[1,2,3,4,5,6])
	#
	# p.set_value('co1', 123)
	#
	# p.utility_co[3] = P.co3 * X.coo3

#
# from larch4.roles import P,X
#
# say = lambda: print('say')
#
# w1 = P("B1")
# w2 = P("B2")
# # print(repr(w1))
# # print(repr(w2))
# # print(repr(w1+w2))
# # print(repr(w1-w2))
# # print(repr(w1*w2))
# # print(repr(w1/w2))
#
# y1 = X("COL1")
# y2 = X("COL2")
# # print(repr(y1))
# # print(repr(y2))
# # print(repr(y1+y2))
# # print(repr(y1-y2))
# # print(repr(y1*y2))
# # print(repr(y1/y2))
#
# lf1= w1*y1
#
# lf1.set_touch_callback(say)
#
# lf2 = w2*y2
#
# # lf1 + w2*y2