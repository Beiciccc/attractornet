import biodivine_aeon as ba
print("BbmModel methods:", [m for m in dir(ba.BbmModel) if not m.startswith('_')][:30])
# try fetch_model for a few ids to find names
for i in ['1','5','10','13','20','30','40','50']:
    try:
        m = ba.BiodivineBooleanModels.fetch_model(i)
        name = getattr(m,'name',None) or (m.name() if callable(getattr(m,'name',None)) else None)
        print(i, '->', type(m).__name__, '| name attr try:', repr(name)[:80])
    except Exception as e:
        print(i, 'fetch_model err', repr(e)[:70])
