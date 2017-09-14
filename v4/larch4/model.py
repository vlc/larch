
import numpy, pandas
from .parameter_collection import ParameterCollection
from .data_collection import DataCollection
from .workspace_collection import WorkspaceCollection

from .nesting.nl_utility import exp_util_of_nests, util_of_nests, exp_inplace_2
from .nesting.nl_prob import conditional_logprob_from_tree_util, elemental_logprob_from_conditional_logprob


class Model(ParameterCollection):

	def __init__(self, *,
				 parameters = (),
				 alts = (),
				 graph = None,
				 datasource = None,
				 **kwarg):

		self._datasource = datasource
		if datasource is not None:
			if isinstance(alts, (numpy.ndarray, pandas.Series, pandas.DataFrame, list, tuple, set)) and len(alts)>0:
				pass # use the override alts
			else:
				alts = datasource.alternative_codes()

		super().__init__(names=parameters, altindex=alts, **kwarg)

		self._graph = graph if graph is not None else self._mnl_graph()
		self.data = None
		self.work = None


	@property
	def graph(self):
		return self._graph

	@graph.setter
	def graph(self, value):
		self._graph = value
		if self.work is not None:
			self.work = WorkspaceCollection(data_coll=self.data, parameter_coll=self, graph=self._graph)


	def load_data(self):
		self.data = DataCollection(
			caseindex=None, altindex=self._altindex, source=self._datasource,
			utility_ca_index=self.utility_ca_vars,
			utility_co_index=self.utility_co_vars,
			quantity_ca_index=self.quantity_ca_vars,
		)
		self.data.load_data(source=self._datasource)
		self.work = WorkspaceCollection(data_coll=self.data, parameter_coll=self, graph=self._graph)


	def _mnl_graph(self):
		from .nesting.tree import NestingTree
		root_id = 0
		while root_id in self._altindex:
			root_id += 1
		t = NestingTree(root_id=root_id)
		t.add_nodes(self._altindex)
		return t


	# SLOWER...
	# def calculate_exp_utility(self):
	# 	self.data._calculate_exp_utility_elemental(self, self.work.exp_util_elementals)
	# 	exp_util_of_nests(self.work.exp_util_elementals, self.work.exp_util_nests, self._graph, self)

	def calculate_exp_utility(self):
		self.data._calculate_utility_elemental(self, self.work.exp_util_elementals)
		util_of_nests(self.work.exp_util_elementals, self.work.exp_util_nests, self._graph, self)

	def calculate_log_probability(self):
		conditional_logprob_from_tree_util(
			self.work.exp_util_elementals,
			self.work.exp_util_nests,
			self._graph,
			self,
			self.work.log_conditional_prob
		)
		elemental_logprob_from_conditional_logprob(
			self.work.log_conditional_prob,
			self._graph,
			self.work.log_prob
		)

	def loglike(self, parameter_values=None):
		self.unmangle()
		if parameter_values is not None:
			self.set_values(parameter_values)
		self.calculate_exp_utility()
		self.calculate_log_probability()
		LL = self.data._calculate_log_like(self.work.log_prob)
		return LL
