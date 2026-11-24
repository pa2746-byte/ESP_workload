"""Interval-based logical region dependencies, without byte-sized Python tables."""
from cuda_source_model import UnsupportedSource


class SerialSourceOrder:
    """A serial native runtime needs only the immediate operation predecessor."""
    def __init__(self):self.next_id=0

    def schedule(self,accesses,stream,blocking=False):
        if stream!='default':raise UnsupportedSource('Native source ordering requires the default stream')
        node=self.next_id;self.next_id+=1
        return dict(id=node,stream=stream,ordered_after=[node-1] if node else [])


class SourceRegionGraph:
    def __init__(self):
        self.nodes=[];self.metadata=[];self.edges=set();self.allocations={};self.regions={}
        self.order=SerialSourceOrder();self.control=set();self.operation=None

    def allocate(self,name,size):
        if name in self.allocations or type(size) is not int or size<=0:raise UnsupportedSource('Invalid source allocation event')
        self.allocations[name]=size;self.regions[name]=[(0,size,None,frozenset())]

    def normalize(self,name,ranges):
        size=self.allocations.get(name,0);result=[]
        for start,end in sorted(ranges):
            if not 0<=start<end<=size:raise UnsupportedSource('Source footprint exceeds allocation')
            if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
            else:result.append([start,end])
        if not result:raise UnsupportedSource('Empty source transfer region')
        return result

    def dependencies(self,name,ranges,writing):
        parents=set()
        for start,end in ranges:
            for a,b,writer,readers in self.regions[name]:
                if a>=end:break
                if b<=start:continue
                if not writing and writer is None:raise UnsupportedSource('Read of uninitialized source region')
                if writer is not None:parents.add(writer)
                if writing:parents.update(readers)
        return parents

    def update(self,name,ranges,node,writing):
        segments=self.regions[name]
        for start,end in ranges:
            result=[]
            for a,b,writer,readers in segments:
                if b<=start or a>=end:result.append((a,b,writer,readers));continue
                if a<start:result.append((a,start,writer,readers))
                result.append((max(a,start),min(b,end),node if writing else writer,
                               frozenset() if writing else readers|{node}))
                if b>end:result.append((end,b,writer,readers))
            segments=result
        merged=[]
        for segment in segments:
            if merged and merged[-1][1]==segment[0] and merged[-1][2:]==segment[2:]:
                prior=merged[-1];merged[-1]=(prior[0],segment[1],prior[2],prior[3])
            else:merged.append(segment)
        self.regions[name]=merged

    def transfer(self,name,ranges,source,destination,stage,writing,parents=()):
        ranges=self.normalize(name,ranges)
        dependencies=self.dependencies(name,ranges,writing)|set(parents)|self.control
        node=len(self.nodes)
        self.nodes.append(dict(id=node,vol=sum(b-a for a,b in ranges),src_comp=source,dest_comp=destination))
        self.metadata.append(dict(id=node,buffer=name,stage=stage,byte_ranges=ranges,
                                  operation_id=self.operation['id'],stream='default',
                                  ordered_after_operations=self.operation['ordered_after']))
        self.edges.update((p,node) for p in dependencies)
        self.update(name,ranges,node,writing)
        return node
