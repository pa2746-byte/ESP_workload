"""Deterministic small fixtures preserving raw AVI pixel and palette semantics."""
import struct
from cuda_source_model import UnsupportedSource


def first_avi_frames(content,count=1,width=None):
    def chunks(data):
        offset=0
        while offset+8<=len(data):
            tag=data[offset:offset+4];size=struct.unpack_from('<I',data,offset+4)[0]
            if offset+8+size>len(data):raise UnsupportedSource('Truncated AVI chunk')
            yield tag,data[offset+8:offset+8+size]
            offset+=8+size+(size&1)
    def chunk(tag,payload):return tag+struct.pack('<I',len(payload))+payload+(b'\0' if len(payload)&1 else b'')
    if content[:4]!=b'RIFF' or content[8:12]!=b'AVI ':raise UnsupportedSource('Fixture needs RIFF AVI')
    end=8+struct.unpack_from('<I',content,4)[0]
    if end>len(content):raise UnsupportedSource('Truncated RIFF extent')
    top=list(chunks(content[12:end]));header=next(p[4:] for t,p in top if t==b'LIST' and p[:4]==b'hdrl')
    fields=list(chunks(header));avih=bytearray(next(p for t,p in fields if t==b'avih'))
    if struct.unpack_from('<I',avih,24)[0]!=1:raise UnsupportedSource('AVI fixture supports one video stream only')
    stream=next(p[4:] for t,p in fields if t==b'LIST' and p[:4]==b'strl');parts=list(chunks(stream))
    strh=bytearray(next(p for t,p in parts if t==b'strh'));fmt=bytearray(next(p for t,p in parts if t==b'strf'))
    original_width,height=struct.unpack_from('<ii',fmt,4);bits=struct.unpack_from('<H',fmt,14)[0]
    if struct.unpack_from('<I',fmt,16)[0]!=0 or bits not in {8,24,32}:raise UnsupportedSource('AVI fixture requires uncompressed pixel storage')
    if count<1:raise UnsupportedSource('AVI fixture needs at least one frame')
    target_width=original_width if width is None else width
    if not 0<target_width<=original_width:raise UnsupportedSource('Invalid AVI fixture crop width')
    old_stride=((original_width*bits+31)//32)*4;new_stride=((target_width*bits+31)//32)*4
    movi=next(p[4:] for t,p in top if t==b'LIST' and p[:4]==b'movi')
    def frames(data):
        for tag,payload in chunks(data):
            if tag in {b'00db',b'00dc'}:yield tag,payload
            elif tag==b'LIST':yield from frames(payload[4:])
    selected=[]
    for tag,payload in frames(movi):
        if len(payload)!=old_stride*abs(height):raise UnsupportedSource('Unexpected raw AVI frame extent')
        if target_width!=original_width:
            length=target_width*(bits//8)
            payload=b''.join(payload[row*old_stride:row*old_stride+length]+bytes(new_stride-length) for row in range(abs(height)))
        selected.append((tag,payload))
        if len(selected)==count:break
    if len(selected)!=count:raise UnsupportedSource('AVI fixture requests unavailable frames')
    struct.pack_into('<I',avih,16,count);struct.pack_into('<I',avih,12,struct.unpack_from('<I',avih,12)[0]|0x10)
    struct.pack_into('<I',avih,32,target_width);struct.pack_into('<I',avih,28,new_stride*abs(height))
    struct.pack_into('<I',strh,28,0);struct.pack_into('<I',strh,32,count);struct.pack_into('<I',strh,36,new_stride*abs(height))
    if len(strh)>=56:struct.pack_into('<h',strh,52,target_width)
    struct.pack_into('<i',fmt,4,target_width);struct.pack_into('<I',fmt,20,new_stride*abs(height))
    stream=b''.join(chunk(t,bytes(strh) if t==b'strh' else bytes(fmt) if t==b'strf' else p) for t,p in parts)
    header=b''.join(chunk(t,bytes(avih) if t==b'avih' else b'strl'+stream if t==b'LIST' and p[:4]==b'strl' else p) for t,p in fields if not(t==b'LIST' and p[:4]==b'odml'))
    body=b'';index=b'';offset=4
    for tag,payload in selected:
        encoded=chunk(tag,payload);body+=encoded;index+=tag+struct.pack('<III',0x10,offset,len(payload));offset+=len(encoded)
    payload=b'AVI '+chunk(b'LIST',b'hdrl'+header)+chunk(b'LIST',b'movi'+body)+chunk(b'idx1',index)
    return b'RIFF'+struct.pack('<I',len(payload))+payload
