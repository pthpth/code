from operator import index
import shutil
import networkx as nx
from itertools import combinations
from networkx import read_graphml
import networkx.algorithms.isomorphism as iso
from pathlib import Path
import math
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
import time

def calculateAllPossibleEdges(numberOfAgents):
    """Use networkx to create a graph with all possible edges between agents"""
    # CompleteDirectedGraph with no self-loops
    # MainGraph = nx.DiGraph(directed=True)
    # MainGraph.add_nodes_from(range(numberOfAgents))
    AllMyEdges = [(i, j) for i in range(numberOfAgents) for j in range(numberOfAgents) if i != j]
    # MainGraph.add_edges_from(AllMyEdges)
    print("There are ", len(AllMyEdges), " possible edges")
    return AllMyEdges

def makeCompleteGraph(numberOfAgents, EdgesList):
    """Use networkx to create a graph with all possible edges between agents"""
    # CompleteDirectedGraph with no self-loops
    MainGraph = nx.DiGraph()
    MainGraph.add_nodes_from(range(numberOfAgents))
    # AllMyEdges = [(i, j) for i in range(numberOfAgents) for j in range(numberOfAgents) if i != j]
    if len(EdgesList) > 0:
        MainGraph.add_edges_from(EdgesList)
    return MainGraph

def makeAllVariantsOfGraph(numberOfAgents):
    """Use networkx to create a graph with all possible edges between agents"""
    # CompleteDirectedGraph with no self-loops
    AllMyEdges = calculateAllPossibleEdges(numberOfAgents)
    EmptyGraph = makeCompleteGraph(numberOfAgents, [])
    CompleteGraph = makeCompleteGraph(numberOfAgents, AllMyEdges)
    # All possible edge combinations
    CurrentGraphNoIsoCollection = [EmptyGraph, CompleteGraph]
    count = 0
    for possibleEdgeNumber in range(len(AllMyEdges)):
        for combo in combinations(AllMyEdges, possibleEdgeNumber):
            NewGraph = makeCompleteGraph(numberOfAgents, combo)
            Checks = [iso.DiGraphMatcher(NewGraph, graphToCompare).is_isomorphic() for graphToCompare in CurrentGraphNoIsoCollection]
            count += 1
            if not any(Checks):
                CurrentGraphNoIsoCollection.append(NewGraph)
    print("Out of ", count, " possible graphs, ", len(CurrentGraphNoIsoCollection), " are unique.")
    return CurrentGraphNoIsoCollection

def DrawSingleGraph(title, graph, ax, color="slateblue", CustomParams=None):
    NewTitle = title
    node_size = 400
    arrowsize = 20
    width = 2
    titleFontSize = 'large'
    if CustomParams is not None: 
        node_size = CustomParams.get("node_size", 400)
        arrowsize = CustomParams.get("arrowsize", 20)
        width = CustomParams.get("width", 2)
        titleFontSize = 'x-large'
        
    if node_size < 40:
        NewTitle = f"{title}"
    else:
        NewTitle = f"{title} ({len(graph.edges)} edges"
        
    # make this axis a bit smaller (scale)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
        
    nx.draw_networkx(
    graph,
    ax=ax,
    with_labels=False,
    node_color=[color],
    edge_color="dimgray",
    node_size=node_size,
    arrowsize=arrowsize,
    linewidths=0.25,
    width=width,
    font_size=10, 
    font_weight='bold',
    pos=nx.planar_layout(graph),
    connectionstyle='arc3, rad = 0.15',
    arrowstyle="-|>",
    # do not print weights 
    # connectionstyle="angle,angleA=-95,angleB=35,rad=10",
    )
    # ax.set_xlim(val * 1.2 for val in ax.get_xlim())
    # ax.set_ylim(val * 1.2 for val in ax.get_ylim())
    ax.set_title(NewTitle, family='monospace', fontsize=titleFontSize) #, bbox=dict(boxstyle="round", alpha=0.2))
    if CustomParams is None: 
        ax.text(
            0,
            0,
            f"{len(graph.edges)} edges",
            # fontsize=12,
            fontsize='medium',
            # fontweight="bold",
            horizontalalignment="center",
            verticalalignment="top",  # Change this line
            bbox={"boxstyle": "square,pad=0.1", "fc": "none"},
        )
    return ax 

def printGraphsOnBigPlot(GraphCollection, gifs_dir, UseDiffColor):
    FullListOfGraphs = [graph for graphList in GraphCollection.values() for graph in graphList] # flatten dictionary
    counted = len(FullListOfGraphs)
    MinimumRowColoumns = math.ceil(math.sqrt(counted))
    
    # Adjust figure size based on the number of subplots
    fig_size = max(15, MinimumRowColoumns * 3)  # Increase figure size for many subplots
    
    with plt.style.context('fivethirtyeight'):
        fig, axes = plt.subplots(MinimumRowColoumns, MinimumRowColoumns, figsize=(fig_size, fig_size))
        # Add background between plots
        fig.patch.set_facecolor('white')
        
        if UseDiffColor:
            # Do per edges count
            MaxEdgesPossible = max(GraphCollection.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, MaxEdgesPossible+1))
        
        TotalCountOfGraphs = sum([len(graphList) for graphList in GraphCollection.values()])
        CustomParams = None
        if TotalCountOfGraphs > 16:
            # Adjust node size and other parameters based on the number of subplots
            node_size = max(10, 400 // (MinimumRowColoumns // 2))
            arrowsize = max(1, 20 // (MinimumRowColoumns // 2))
            width = max(0.1, 2 / (MinimumRowColoumns // 2))
            CustomParams = {"node_size": node_size, "arrowsize": arrowsize, "width": width, "titleFontSize": 'xx-small'} 
        
        GraphsDone = 0
        for edges, graphList in GraphCollection.items():
            for count, graph in enumerate(graphList):
                x = GraphsDone // MinimumRowColoumns
                y = GraphsDone % MinimumRowColoumns
                
                ax = axes[x][y] if MinimumRowColoumns > 1 else axes[y]
                ax.set_label(f"Type {edges}E-{count}")
                if UseDiffColor:
                    color = colors[edges]
                else:
                    color = "slateblue"
                ax = DrawSingleGraph(f"Type {edges}E-{count}", graph, ax, color=color, CustomParams=CustomParams)
                GraphsDone += 1
        
        # Remove empty subplots
        for i in range(GraphsDone, MinimumRowColoumns**2):
            x = i // MinimumRowColoumns
            y = i % MinimumRowColoumns
            if MinimumRowColoumns > 1:
                fig.delaxes(axes[x][y])
            elif i < len(axes):
                fig.delaxes(axes[i])
        
        fig.tight_layout()
    
    save_path = None
    if os.path.exists(gifs_dir):
        save_path = f"{gifs_dir}/SI_All_Graphs_Variants({counted}).png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.clear()
    plt.clf()
    plt.cla()
        
    return save_path

def sortedDictionaryFromList(MyList):
    """Takes a list of graphs and returns a dictionary with the number of edges as the key"""
    if isinstance(MyList, dict):
        return MyList # already a dictionary
    
    MyDict = {}
    for graph in MyList:
        NumberOfEdges = len(graph.edges)
        if NumberOfEdges in MyDict:
            MyDict[NumberOfEdges].append(graph)
        else:
            MyDict[NumberOfEdges] = [graph]
    return MyDict
    
def prepUniqueSIGraphVariants(numberOfAgents, gifs_dir, UseDiffColor=True):
    """Makes a list of Digraphs with all possible edges between agents"""
    GraphCollection = makeAllVariantsOfGraph(numberOfAgents)
    GraphCollection = sortedDictionaryFromList(GraphCollection)
    # print("There are " + str(len(GraphCollection)) + " unique non-isometric graphs")
    save_path = printGraphsOnBigPlot(GraphCollection, gifs_dir, UseDiffColor=UseDiffColor)
    return GraphCollection, save_path

def get_factor_score(graph, num_agents):
    """Get the score of the graph"""
    # NumberOfEdges, G_ID = graphTypeCoords
    # graph = self.GraphCollection[NumberOfEdges][G_ID]
    DoubleFactor = 1 + (num_agents * (num_agents - 1)) / 2
    
    AllEdges = graph.edges
    MyDoubleCount = 0
    MySingleCount = 0
    # for all the edges in this graph, check if there is another edge in opposite direction, if so remove from MySet 
    for edge in AllEdges:
        if (edge[1], edge[0]) in AllEdges:
            MyDoubleCount += 1
        else:
            MySingleCount += 1
    MyDoubleCount = int(MyDoubleCount / 2)
    MySingleCount = int(MySingleCount)
    MyScore = MySingleCount + (MyDoubleCount * DoubleFactor)
    return int(MyScore), (MySingleCount, MyDoubleCount)

    
class GraphHandler:
    def __init__(self, numberOfAgents, gifs_dir="Gifs", saveGraphsToLocalStores=False, LoadGraphsFromLocalStores=False):
        self.internal_counter_of_refreshes = 0
        self.numberOfAgents = numberOfAgents
        Path(gifs_dir).mkdir(parents=True, exist_ok=True)
        self.uniqueNoEdges = None
        self.CutOffPointsForGraphTypes = None
        self.GraphTypes = None
        self.statement = None 
        self.GraphCollection = None
        self.GraphCollectionInfo = None
        self.GraphBondInfo = None
        self.DifferentGraphsByScore = None
        self.MapScoreToIndex = None
        self.gifs_dir = gifs_dir
        self.HaveNotSavedBefore = True # check if we have saved before
        self.saveGraphsToLocalStores = saveGraphsToLocalStores
        self.LoadGraphsFromLocalStores = LoadGraphsFromLocalStores
    
        numberEdgesPossible = numberOfAgents * (numberOfAgents - 1)
        self.numberEdgesPossible = numberEdgesPossible
        if numberOfAgents <= 3:
            print("Since only 3 agents, we can calculate all possible graphs")
            # save_path can be returned if we want to use it later i.e wandb
            self.GraphCollection, save_path = prepUniqueSIGraphVariants(numberOfAgents, gifs_dir, UseDiffColor=True)
            # self.GraphCollection = self.sortedDictionaryFromList(GraphCollection)
            self.refresh()
        else: 
            # Too many agents to calculate all possible graphs # the total number of edges can be are N(N-1) where N is the number of agents
            self.GraphCollection = {i: [] for i in range(numberEdgesPossible+1)} #add one as zero index and zero is possible
            
            MyRootDir = Path(__file__).parent
            self.yaml_path_folder = os.path.join(MyRootDir, f"graph_store-{numberOfAgents}_agents")
            
            if LoadGraphsFromLocalStores:
                self.load_graphs_from_yaml()
                
            
        
        
    def load_graphs_from_yaml(self):
        start_time = time.time()
        loadCount = 0
        print("Loading graphs from local store")
        # if folder exist iterate over all files and load them into the graph collection
        if os.path.exists(self.yaml_path_folder):
            for file in os.listdir(self.yaml_path_folder):
                # if graphs exist, load into the graph collection
                if file.endswith(".graphml"):
                    G = read_graphml(f"{self.yaml_path_folder}/{file}")
                    NumberOfEdges = len(G.edges)
                    # load the file and add to graph collection 
                    self.GraphCollection[NumberOfEdges].append(G)
                    loadCount += 1
        self.refresh()
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Loaded {loadCount} graphs from local store in {elapsed_time:.2f} seconds")
            
    def save_graphs_to_yaml(self):
        start_time = time.time()
        if os.path.exists(self.yaml_path_folder) and self.HaveNotSavedBefore and not self.LoadGraphsFromLocalStores:
            # delete the folder and recreate it    
            shutil.rmtree(self.yaml_path_folder)  
            self.HaveNotSavedBefore = False
        # replace with new set 
        # create the folder
        GraphCount = 0
        Path(self.yaml_path_folder).mkdir(parents=True, exist_ok=True)
        for NumberOfEdges, graphList in self.GraphCollection.items():
            for count, graph in enumerate(graphList):
                # if file does not already exist 
                fileName = f"{self.yaml_path_folder}/Type_{NumberOfEdges}E-{count}.graphml"
                if not os.path.exists(fileName):
                    nx.write_graphml(graph, f"{self.yaml_path_folder}/Type_{NumberOfEdges}E-{count}.graphml")
                GraphCount += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Saved all {GraphCount} graphs to local store in {elapsed_time:.2f} seconds")

    def produce_big_plot(self):
        save_path = printGraphsOnBigPlot(self.GraphCollection, self.gifs_dir, UseDiffColor=True)
        return save_path

    def refresh(self):
        """Run everytime you get a new graph"""
        self.setup_color()
        self.analyzeNumberOfEdges()
        self.list_statement(self.GraphCollection)
        self.GraphTypes = sum([len(graphList) for graphList in self.GraphCollection.values()])
        self.internal_counter_of_refreshes += 1
        if self.internal_counter_of_refreshes % 10 == 0 and self.saveGraphsToLocalStores:
            print("Refresh Graphes Stored")
            self.save_graphs_to_yaml()
            
            
        
    def get_statement(self, graphTypeCoords, addScore=False):
        """Get the statement of the graph"""
        NumberOfEdges, G_ID = graphTypeCoords
        title = f"Type {NumberOfEdges}E-{G_ID}"
        if self.GraphCollectionInfo is not None and self.GraphCollectionInfo.get(title, None) is not None:
            score = self.GraphCollectionInfo.get(title)
            bondCounts = self.GraphBondInfo.get(score)
        else:
            graph = self.GraphCollection[NumberOfEdges][G_ID]
            score, bondCounts = get_factor_score(graph, self.numberOfAgents)
            
        Single, Double = bondCounts
        proper_title = f"{NumberOfEdges}E-{Single}-{Double}"
        if addScore:
            return proper_title + " ({score})"
        return proper_title
        
        
    def list_statement(self, GraphCollection): 
        Info = {}
        BondInfo = {}
        Titles = {}

        for NumberOfEdges, graphList in GraphCollection.items():
            for count, graph in enumerate(graphList):
                title = f"Type {NumberOfEdges}E-{count}" # key is edges, count is which graph this is
                score, bondCounts = get_factor_score(graph, self.numberOfAgents)
                
                Single, Double = bondCounts
                proper_title = f"Type {NumberOfEdges}E-{Single}-{Double} ({score})"
                Titles[proper_title] = score
                
                Info[title] = score
                BondInfo[score] = bondCounts
                
        # sort by score and produce a list
        Info = {k: v for k, v in sorted(Info.items(), key=lambda item: item[1])}
        self.GraphCollectionInfo = Info
        
        Titles = {k: v for k, v in sorted(Titles.items(), key=lambda item: item[1])}
        MyList = Titles.keys()
        self.statement = MyList
        
        # Sort bond info by score
        BondInfo = {k: v for k, v in sorted(BondInfo.items(), key=lambda item: item[0])}
        self.GraphBondInfo = BondInfo
        
        UniqueScores = list(set(Info.values()))
        self.DifferentGraphsByScore = len(UniqueScores)
        self.MapScoreToIndex = {score: index for index, score in enumerate(UniqueScores)}

    def setup_color(self):
        # MaxEdgesPossible = max([len(graph.edges) for graph in self.GraphCollection])
        MaxEdgesPossible = max(self.GraphCollection.keys())
        self.colormap = plt.cm.viridis(np.linspace(0, 1, MaxEdgesPossible+1))
        
    def get_score(self, graphTypeCoords):
        """Get the score of the graph"""
        NumberOfEdges, G_ID = graphTypeCoords
        
        if NumberOfEdges == -1: # Cannot find issue
            return -1
        
        Title = f"Type {NumberOfEdges}E-{G_ID}"
        if self.GraphCollectionInfo is not None and self.GraphCollectionInfo.get(Title, None) is not None:
            score = self.GraphCollectionInfo.get(Title)
            return score, self.GraphBondInfo.get(score)
        else:
            graph = self.GraphCollection[NumberOfEdges][G_ID]
            return get_factor_score(graph, self.numberOfAgents)
        
    def get_full_score(self, graphTypeCoords):
        NumberOfEdges, G_ID = graphTypeCoords
        
        if NumberOfEdges == -1: # Cannot find issue
            return -1
        
        graph = self.GraphCollection[NumberOfEdges][G_ID]
        return get_factor_score(graph, self.numberOfAgents)
            
        
    def check_which_graph(self, graphToCheck):
        """Check which graph the input graph is, by number of edges first, then by isomorphism"""
        NumberOfEdges = len(graphToCheck.edges)
        GraphListToCheck = self.GraphCollection[NumberOfEdges]

        for x, graph in enumerate(GraphListToCheck):
            if iso.DiGraphMatcher(graph, graphToCheck).is_isomorphic():
                return (NumberOfEdges, x) # unique ID is a vector of number of edges and index in the list
        
        # If not found, add to collection
        # if self.numberOfAgents <= 3:
        #     print("Graph type not within collection - BUG?")
        #     print("Graph edges: ", graphToCheck.edges)
        #     return (-1, -1)
        # else:
        #     self.GraphCollection[NumberOfEdges].append(graphToCheck)
        #     self.refresh() 
        #     return (NumberOfEdges, len(self.GraphCollection[NumberOfEdges]) - 1) # since zero indexed
        
        print(f"Graph type not within collection - Adding now {NumberOfEdges}E-{len(GraphListToCheck)}")
        self.GraphCollection[NumberOfEdges].append(graphToCheck)
        self.refresh() 
        return (NumberOfEdges, len(self.GraphCollection[NumberOfEdges]) - 1) # since zero indexed
    
    def draw_graph_type(self, graphType, ax, CustomParams=None):
        """Draw the graph of the given type"""
        GraphEdges, IndexOfGraph = graphType
        MyGraphList = self.GraphCollection.get(GraphEdges, None)
        if MyGraphList is None:
            print("Graph type with these edges not found")
            return ax
        try:
            MySpecificGraph = MyGraphList[IndexOfGraph]
        except IndexError:
            MySpecificGraph = None
            print("Graph not found, index error?")
            return ax
        
        ColorToUse = self.colormap[GraphEdges]
        title = f"Type {GraphEdges}E-{IndexOfGraph}"
        ax = DrawSingleGraph(title, MySpecificGraph, ax, color=ColorToUse, CustomParams=CustomParams)

        return ax
    
    def analyzeNumberOfEdges(self):
        """Analyze the number of edges in the graph collection"""
        uniqueNoEdges = len(self.GraphCollection.keys())
        IndexesOfFirstOccurance = [0]
        for key in self.GraphCollection.keys():
            IndexesOfFirstOccurance.append(IndexesOfFirstOccurance[-1] + len(self.GraphCollection[key]))
        # Should be 0,1,2,2,2,2,3,3,3,3,4,4,4,4,5,6
        # uniqueNoEdges, IndexesOfFirstOccurance = np.unique(NumberOfEdges, return_index=True)
        # self.uniqueNoEdges = max(uniqueNoEdges)
        # self.CutOffPointsForGraphTypes = list(IndexesOfFirstOccurance)
        
        self.uniqueNoEdges = uniqueNoEdges
        self.CutOffPointsForGraphTypes = IndexesOfFirstOccurance

