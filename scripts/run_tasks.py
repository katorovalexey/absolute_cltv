import logging.config

import networkx as nx

from cltv_tools.config.logging import DEFAULT_CONF
from cltv_tools.config.spark_preconf import spark_preconf
from cltv_tools.utils.kerberos import kinit
from cltv_tools.utils.luigi import deps_graph
from absolute_cltv.total_pnl.tasks.modelling.scoring import ScoringCombinePredictions

logging.config.dictConfig(DEFAULT_CONF)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    kinit(source='hdfs')
    spark_preconf()
    task = ScoringCombinePredictions()

    # deps_graph(task, plotly=False, fig_path='docs/images/deps_graph.png')
    # g = deps_graph(task, plotly=True, fig_path='reports/deps_graph.html')

    g = deps_graph(task)
    for node in g.nodes:
        del g.nodes[node]['task']
    nx.write_gml(g, 'reports/deps_graph.gml')
