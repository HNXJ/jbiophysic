import jaxley as jx
from jbiophysics.channels.hh import SafeHH

def build_pyramidal_cell():
    cell = jx.Cell()
    cell.insert(SafeHH())
    return cell

def build_pv_cell():
    cell = jx.Cell()
    cell.insert(SafeHH())
    return cell

def build_sst_cell():
    cell = jx.Cell()
    cell.insert(SafeHH())
    return cell

def build_vip_cell():
    cell = jx.Cell()
    cell.insert(SafeHH())
    return cell
