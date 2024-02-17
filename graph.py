from queue import Queue
import numpy as np
import sys
from copy import deepcopy

#All edges in graphs are undirected and obey the rule: v1<v2
#For trees it is important, since we want to establish a parent-child relation
def normalize_edge(e):
  if e[0]>e[1]:
    return e[::-1]
  else:
    return e

class Node:
  def __init__(self,_id):
    self.children=[]
    self.id=_id

  def __eq__(self,other):
    return self.id==other.id

  def add_child(self,child):
    if not child in self.children:
      self.children.append(child)

  def __str__(self):
    dest="ID: "+str(self.id)+"\n"
    id_vec=[str(x.id) for x in self.children]
    dest=dest+"Children: "+str(id_vec)#+"\n"
    return dest
    
class Graph:
  def __init__(self):
    self.nodevec=[]
    self.edgemat=np.zeros((0,0))

  def add_node(self,node):
    self.nodevec.append(node)
    self.edgemat=np.pad(self.edgemat,((0,1),(0,1)),'constant', constant_values=(0, 0))
    return(len(self.nodevec)-1)

  def add_weighted_edge_sym(self,e,weight):
    eind=tuple(map(self.id2ind,e))
    self.edgemat[eind]=weight 
    self.edgemat[eind[::-1]]=weight 

  def id2ind(self,nid):
    return self.nodevec.index(nid)

  def __str__(self):
    dest=""
    dest+=str(self.nodevec)
    dest+="\n"
    dest+=str(self.edgemat)
    return dest

  def to_edgelist(self):
    dest=set()
    for nid in self.nodevec:
      edgelist_node=self.get_edgelist(nid)
      for e in edgelist_node:
        dest.add(e)
    return list(dest)

  def get_edgelist(self,nid):
    dest=[]
    ni,nj=self.edgemat.shape
    indi=self.id2ind(nid)
    for j in range(nj):
      if self.edgemat[indi,j]!=0:
        dest.append(normalize_edge((nid,self.nodevec[j])))
    return dest

  def get_neighbors(self,nid):
    dest=set()
    edgelist=self.get_edgelist(nid)
    for v1,v2 in edgelist:
      if v1==nid:
        dest.add(v2)
      else:
        dest.add(v1)
    return list(dest)

  def remove_vertex(self,nid):
    nind=self.id2ind(nid)
    self.nodevec.remove(nid)
    self.edgemat=np.delete(self.edgemat,nind,0)
    self.edgemat=np.delete(self.edgemat,nind,1)

  def get_weight(self,e):
    v1,v2=e
    return self.edgemat[self.id2ind(v1),self.id2ind(v2)]

  def calculate_weight_path(self,path):
    dest=0.
    for e in path:
      dest+=self.get_weight(e)
    return dest

#TODO: We could make this a bit more reasonable by checking that a tree has no cycles
class Tree:
  def __init__(self,root):
    if type(root) is Node:
      self.root=root
    else:
      self.root=Node(root)

  def dfs(self,node,func,*args):
    for child in node.children:
      self.dfs(child,func,*args)
    func(node,*args)

  def bfs(self,node,func,*args):
    q=Queue()
    q.put(node)
    while not q.empty():
      el=q.get()
      func(el,*args)
      for item in el.children:
        q.put(item)

  def find(self,nid):
    q=Queue()
    q.put(self.root)
    while not q.empty():
      el=q.get()
      if el.id==nid:
        return el
      for item in el.children:
        q.put(item)

  def bfs_until(self,sid,func,*args):
    q=Queue()
    q.put(self.root)
    while not q.empty():
      el=q.get()
      func(el,*args)
      if el.id==sid:
        return el
      for item in el.children:
        q.put(item)

  def get_path(self,node,nid,dest):
    dest.append(node.id)
    if node.id==nid:
      return dest
    if len(node.children)==0:
      return 
    for child in node.children:
      copy_dest=deepcopy(dest)
      sol=self.get_path(child,nid,copy_dest)
      if sol is not None:
        return sol
    
  def _add_node_to_list(self,node,nodelist):
    nodelist.append(node)

  def _add_edge_to_list(self,node,edgelist):
    for child in node.children:
      edgelist.append((node.id,child.id))

  def depth(self,item):
    path=[]
    if type(item) is Node:
      path=self.get_path(self.root,item.id,path)
    else:
      path=self.get_path(self.root,item,path)
    return len(path)-1

  def to_nodelist(self):
    dest=[]
    self.dfs(self.root,self._add_node_to_list,dest)
    return dest

  def to_edgelist(self):
    dest=[]
    self.dfs(self.root,self._add_edge_to_list,dest)
    return dest
  
  def add_edge(self,e):
    #The order of the edge is important!
    #The parent comes first
    self.add_node_at_id(e[0],Node(e[1]))

  def __contains__(self,item):
    if type(item) is Node:
      return self.find(item.id) is not None
    else:
      return self.find(item) is not None

  def add_node_at_id(self,node_id,child):
    node=self.find(node_id)  
    node.children.append(child)

  def add_node_at_node(self,node,child):
    node.children.append(child)

class Forest:
  def __init__(self):
    self.treevec=[]

  def add_tree(self,tree):
    self.treevec.append(tree)

  def __contains__(self,item):
    for tree in self.treevec:
      if item in tree:
        return True
    return False

  def get_tree(self,tree_ind):
    return self.treevec[tree_ind]

  def to_nodelist(self):
    dest=[]
    for tree in self.treevec:
      dest.append(*tree.to_nodelist())
    return dest

  def get_containing_tree(self,node):
    for tree in self.treevec:
      if node in tree:
        return tree
    return None

  def get_root(self,node):
    tree=self.get_containing_tree(node)
    if tree is not None:
      return tree.root
    else:
      return None

def print_id(node):
  print(node.id)

class Path():
  def __init__(self):
    self.set=set([])

  def add_edge(self,v1,v2=None):
    e=None
    if v2 is None:
      if type(v1) is tuple and len(v1)==2:
        e=normalize_edge(v1)
    else:
      if type(v1) is int and type(v2) is int:
        e=normalize_edge((v1,v2))
    if e is None:
      raise TypeError("Cannot enter an edge for parameters of type "+type(v1).__name__+" and "+type(v2).__name__)
    self.set.add(e)

  def contains_vertex(self,v):
    for v1,v2 in self.set:
      if v1==v or v2==v:
        return True
    return False

  def __len__(self):
    return len(self.set)

  def __contains__(self,item):
    item_rev=item[::-1]
    return item in self.set or item_rev in self.set

  def __str__(self):
    return str(self.set)
    
  def __iter__(self):
    return self.set.__iter__()

  def __add__(self,other):
    if type(other) is Path:
      dest=Path()
      dest.set=self.set.union(other.set)
      return dest
    else:
      raise TypeError("Path and "+type(other).__name__+" cannot be added.")

  def __radd__(self,other):
    if type(other) is Path:
      dest=Path()
      dest.set=self.set.union(other.set)
      return dest
    else:
      raise TypeError("Path and "+type(other).__name__+" cannot be added.")

  def add_vertexlist(self,vertexlist):
    for i in range(len(vertexlist)-1):
      e=(vertexlist[i],vertexlist[i+1])
      self.set.add(normalize_edge(e))

  def add_edgelist(self,edgelist):
    for e in edgelist:
      self.set.add(normalize_edge(e))

  def substitute(self,old_edges,new_edges):
    for e in old_edges:
      self.set.remove(e)
    for e in new_edges:
      self.set.add(e)

class Matching(Path):
  def augment(self,path):
    self.set.symmetric_difference_update(path.set)

  def add_edge(self,v1,v2):
    for e in self.set:
      if v1 in e or v2 in e:
        print("Edge not inserted. A vertex cannot be inserted twice into a matching",file=sys.stderr)
        return
    Path.add_edge(self,v1,v2)

  def get_matching_vertex(self,v):
    for v1,v2 in self.set:
      if v1==v:
        return v2
      if v2==v:
        return v1

class Blossom:
  def __init__(self,edgevec,graph):
    self.edgevec=edgevec
    self.graph=graph
    self.blossom_graph=None
    self.vertexlist=None

  def get_external_vertex_dict(self):
    #Find all vertices that are neighbors of vertices in the blossom
    vertexlist=self.get_vertexlist()
    neighbors={}
    for vid in vertexlist:
      neighbors[vid]=self.graph.get_neighbors(vid)
    #Filter out only the ones that are not in the blossom itself
    external_vertices={}
    for vid,arr in neighbors.items():
      external=[x for x in arr if x not in vertexlist]
      external_vertices[vid]=external
    return external_vertices

  def get_edges_to_environment(self):
    vertexdict=self.get_external_vertex_dict()
    dest=[]
    for key,arr in vertexdict.items():
      for v in arr:
        e=(key,v)
        dest.append(normalize_edge(e))
    return dest

  def get_vertexlist(self):
    #Get all indices of vertices that are in the blossom
    if self.vertexlist is None:
      vertexset=set()
      for v1,v2 in self.edgevec:
        vertexset.add(v1)
        vertexset.add(v2)
      self.vertexlist=list(vertexset)
    return self.vertexlist

  def contract(self,matching=None):
    vertexlist=self.get_vertexlist()
    external_vertices=self.get_external_vertex_dict()
    #Copy the graph and delete everything that is in the blossom and 
    #reconnect the edges to the new contracted vertex
    g = deepcopy(self.graph)
    max_node=max(g.nodevec)
    self.blossom_id=max_node+1
    blossom_ind=g.add_node(self.blossom_id)
    for key in external_vertices:
      ext_neighbors=external_vertices[key]
      for vertex in ext_neighbors:
        g.add_weighted_edge_sym((g.id2ind(vertex),blossom_ind),g.edgemat[g.id2ind(key),g.id2ind(vertex)])
    for vid in vertexlist:
      g.remove_vertex(vid)
    self.contracted_graph=g
    return g 
  
  def get_blossom_neighbor(self,nid):
    vertexlist=self.get_vertexlist()
    if nid in vertexlist:
      return nid
    vertexdict=self.get_external_vertex_dict()
    for key,item in vertexdict.items():
      for ext_id in item:
        if ext_id==nid:
          return key

  def get_blossom_graph(self):
    if self.blossom_graph is None:
      dest=Graph()
      vertexlist=self.get_vertexlist()
      for vertex in vertexlist:
        dest.add_node(vertex)
      for e in self.edgevec:
        weight=self.graph.get_weight(e)
        dest.add_weighted_edge_sym(e,weight)
      self.blossom_graph=dest
    return self.blossom_graph

  def get_even_path(self,graph,v1,v2):
    #This function works only on a ring of odd length(e.g. a blossom)
    nodevec=[v1]
    nodelist=graph.get_neighbors(v1)
    while len(nodelist)>0:
      v=nodelist.pop()
      nodevec.append(v)
      if v==v2:
        break
      neighbors=graph.get_neighbors(v)
      for new_v in neighbors:
        if new_v not in nodevec:
          nodelist.append(new_v)
    if len(nodevec)%2==0:
      #The path is the one that has an odd number of edges: We have to take all other edges
      unordered_vertexlist = [x for x in graph.nodevec if x not in nodevec or x == v1 or x == v2]
      nodevec=[v1]
      v=v1
      while v != v2:
        neighbors=graph.get_neighbors(v)
        for neighbor in neighbors:
          if neighbor in unordered_vertexlist and neighbor not in nodevec:
            nodevec.append(neighbor)
            v=neighbor
    dest=Path()
    for ind in range(len(nodevec)-1):
      dest.add_edge(nodevec[ind],nodevec[ind+1])
    return dest

  def expand(self,path):
    dest=deepcopy(path)
    neighbors=self.contracted_graph.get_neighbors(self.blossom_id)
    connection_vertices=[]
    for nid in neighbors:
      connection_vertices.append(self.get_blossom_neighbor(nid))
    assert(len(connection_vertices)==2)
    blossom_graph=self.get_blossom_graph()
    path_new=self.get_even_path(blossom_graph,*connection_vertices)
    #Get rid of the old vertices that connect the blossom to the rest of the graph and enter the internal edges of the blossom
    dest.substitute(self.contracted_graph.get_edgelist(self.blossom_id),path_new)
    #Add the connections from the environment to the blossom
    external_edges=self.get_edges_to_environment()
    dest.add_edgelist(external_edges)
    return dest


def test_tree():
  root=Node(0)
  t=Tree(root)
  n1=Node(1)
  n1.add_child(Node(3))
  root.add_child(n1)
  root.add_child(Node(2))
  print("DFS")
  t.dfs(t.root,print_id)
  print("BFS")
  t.bfs(t.root,print_id)
  nodelist=[str(x) for x in t.to_nodelist()]
  print("Nodelist: ",nodelist)
  print("Edgelist: ",t.to_edgelist())
  print("Find")
  node=t.find(3)
  print(node)
  print("Depth of node 0:",t.depth(t.root))
  print("Depth of node 1:",t.depth(n1))
  print("Depth of node 2:",t.depth(Node(2)))
  print("Depth of node 3:",t.depth(Node(3)))
  path=t.get_path(t.root,3,[])
  print(path)

def test_path():
  p1=Path()
  p1.add_edge(0,1)
  p1.add_edge(6,7)
  print("Path: ",p1)
  p2=Path()
  p2.add_edge(2,3)
  p3=p1+p2
  print("Added path:",p3)
  m=Matching()
  m.add_edge(0,1)
  m.add_edge(2,3)
  m.add_edge(4,5)
  print("Matching vertex to 4: ",m.get_matching_vertex(4))
  for e in m:
    print(e)
  print("Contains vertex 4:",m.contains_vertex(4))
  print("Matching: ",m)
  m.augment(p1)
  print("Augmented Matching: ",m)

def test_graph():
  g=Graph()
  g.add_node(0)
  g.add_node(1)
  g.add_node(2)
  g.add_node(3)
  g.add_node(4)
  g.add_weighted_edge_sym((0,1),1)
  g.add_weighted_edge_sym((1,2),1)
  g.add_weighted_edge_sym((2,3),1)
  g.add_weighted_edge_sym((1,4),1)
  print(g)
  print(g.get_edgelist(1))

def test_blossom():
  g=Graph()
  g.add_node(0)
  g.add_node(1)
  g.add_node(2)
  g.add_node(3)
  g.add_node(4)
  g.add_node(5)
  g.add_node(6)
  g.add_weighted_edge_sym((0,1),1)
  g.add_weighted_edge_sym((1,2),1)
  g.add_weighted_edge_sym((2,3),1)
  g.add_weighted_edge_sym((3,4),1)
  g.add_weighted_edge_sym((4,5),1)
  g.add_weighted_edge_sym((5,1),1)
  g.add_weighted_edge_sym((4,6),1)
  print(g)
  path=[(1,2),(2,3),(3,4),(4,5),(5,1)]
  b=Blossom(path,g)
  g_contracted=b.contract(None)
  print(g_contracted)
  p=Path()
  p.add_edge(0,7)
  p.add_edge(7,6)
  print(b.expand(p))

def main():
  #test_tree()
  #print()
  #test_path()
  #print()
  #test_graph()
  #print()
  test_blossom()

if __name__=="__main__":
  main()
