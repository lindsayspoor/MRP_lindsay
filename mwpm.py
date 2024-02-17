#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np
from graph import Node,Tree,Forest,Path,Graph,Matching,Blossom

class BlossomMatcher:
  def __init__(self):
    pass

  def find_augmenting_path(self,graph,matching):
    #Init for the function
    forest=Forest()
    marked_vertices=[]
    marked_edges=[]
    
    for e in (matching or []):
      marked_edges.append(e)
    for v in graph.nodevec:
      if not matching or (matching and not matching.contains_vertex(v)):
        t=Tree(v)
        forest.add_tree(t)
    #Loop over all unmarked vertices v that have even distance to root(v)
    forest_has_changed=False
    ind=0
    nodelist=forest.to_nodelist()
    while ind < len(nodelist):
      v=nodelist[ind]
      vtree=forest.get_containing_tree(v.id)
      depth=vtree.depth(v)
      if (v.id not in marked_vertices) and depth%2==0:
      #Loop over all unmarked edges (v,w)
        edgevec=graph.get_edgelist(v.id)
        for e in edgevec:
          if e not in marked_edges:
            w=e[0] if e[1]==v.id else e[1]
            if Node(w) not in forest:
              #w is matched so add e and w's matched edge to the forest
              x=matching.get_matching_vertex(w)
              tree=forest.get_containing_tree(v)
              tree.add_edge((v.id,w))
              tree.add_edge((w,x))
            else:
              wtree=forest.get_containing_tree(w)
              if wtree.depth(w)%2==0:
                if forest.get_root(Node(w))!=forest.get_root(v):
                  #Report an augmenting path
                  path_v=vtree.get_path(vtree.root,v.id,[])
                  path_w=wtree.get_path(wtree.root,w,[])
                  path_total=[*path_v,*path_w[::-1]]
                  dest=Path()
                  dest.add_vertexlist(path_total)
                  return dest
                else:
                  # Contract a blossom in G and look for the path in the contracted graph.
                  path_vw=vtree.get_path(v,w,[])
                  b=Blossom(path_vw,graph)
                  graph_contr,matching_contr=b.contract(matching)
                  path=self.find_augmenting_path(graph_contr,matching_contr)
                  dest=b.expand(path)
                  return dest
            marked_edges.append(e)
        marked_vertices.append(v.id)
      #We have to recompute this list after each iteration
      if forest_has_changed:
        nodelist=forest.to_nodelist()
        ind=0
      else:
        ind+=1

  def find_maximum_matching(self,graph,matching):
    if matching is None:
      matching=Matching()
    path=self.find_augmenting_path(graph,matching)
    if path is not None:
      matching.augment(path)
      return self.find_maximum_matching(graph,matching)
    else:
      return matching

  def find_minimum_weight_maximum_matching(self,graph):
    edgelist=graph.to_edgelist()
    matchingvec=[]
    for edge in edgelist:
      primer=Matching()
      primer.add_edge(*edge)
      matching=self.find_maximum_matching(graph,primer) 
      weight=graph.calculate_weight_path(matching)
      matchingvec.append([matching,weight])
    matchingvec_sorted=sorted(matchingvec,key=lambda x:(len(x[0]),-x[1]))
    return matchingvec_sorted[-1]
