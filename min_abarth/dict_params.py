params = {"M":18,"L":17}
import abarth

def abarth_fit(**kwargs):
	return abarth.Abarth(
		kwargs.get('M',200),kwargs.get('L',1),
		kwargs.get("N_sweeps",40),kwargs.get("Nmin",1),kwargs.get("Ncutpoints",100), #CHANGE 
		kwargs.get("alpha",0.95),kwargs.get("beta",1.25 ),kwargs.get("tau":0.3), #CHANGE!))
		kwargs.get("burnin":15),kwargs.get("max_depth_num":250),kwargs.get("draw_sigma":False),
		kwargs.get("kap":16),kwargs.get("s":4),kwargs.get("verbose",False),
		kwargs.get("m_update_sigma",False),kwargs.get("draw_mu":False),kwargs.get("parallel",False)
		)	

print abarth_fit(**params).get_M()

params = {}

print abarth_fit(**params).get_M()


