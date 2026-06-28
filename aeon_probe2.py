import biodivine_aeon as ba
bn = ba.BiodivineBooleanModels.fetch_network('3')
v0 = bn.variables()[5]
print("predecessors:", bn.predecessors(v0))
regs = bn.regulations()
print("one regulation dict:", regs[0])
fn = bn.get_update_function(v0)
print("fn:", str(fn)[:100])
print("UpdateFunction methods:", [m for m in dir(fn) if not m.startswith('_')])
# try evaluate / support vars
import biodivine_aeon as ba2
print("support_set?" , hasattr(fn,'support_set'))
try:
    print("support_variables:", fn.support_variables())
except Exception as e:
    print("supp err", repr(e)[:60])
# attempt evaluation via a valuation dict
try:
    print("eval all-true:", fn.evaluate({v: True for v in bn.variables()}))
except Exception as e:
    print("eval err", repr(e)[:80])
