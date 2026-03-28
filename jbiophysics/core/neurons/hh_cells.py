import jaxley as jx

def build_pyramidal_rs():
    cell = jx.Cell([jx.Compartment(), jx.Compartment()], parents=[-1, 0])
    cell.radius, cell.length = 1.0, 100.0
    cell.insert(jx.channels.HH())
    return cell

def build_pv_fs():
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.radius, cell.length = 1.0, 10.0
    cell.insert(jx.channels.HH())
    return cell
