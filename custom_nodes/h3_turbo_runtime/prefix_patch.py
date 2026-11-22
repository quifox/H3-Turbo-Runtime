def install_prefix24():
    """Patch the pinned ComfyUI loader for 24-layer text features; reject unknown layouts."""
    import inspect
    from comfy import sd
    anchor="    parameters = 0\n    for c in clip_data:\n"
    source=inspect.getsource(sd.load_text_encoder_state_dicts)
    if source.count(anchor)!=1:
        raise RuntimeError("unexpected Comfy text-encoder loader revision")
    replacement="""    if model_options.get('h3_prefix_experiment'):
        assert len(clip_data) == 1 and te_model == TEModel.QWEN3VL_4B and clip_type == CLIPType.KREA2
        result = {}
        for name,value in clip_data[0].items():
            if name.startswith('model.layers.') and int(name.split('.')[2]) >= 24:
                continue
            if not name.startswith(('model.norm.','model.lm_head.')):
                result[name] = value
        assert any(name.startswith('model.layers.23.') for name in result)
        clip_data[0] = result

"""
    exec(compile(source.replace(anchor,replacement+anchor),"<prefix24>","exec"),sd.__dict__)
    load=sd.load_clip
    def load_prefix(*args,**kwargs):
        options=dict(kwargs.get('model_options',{}))
        options.update(h3_prefix_experiment=True,qwen3vl_4b_model_config={
            'num_hidden_layers':24,'final_norm':False,'lm_head':False})
        kwargs['model_options']=options
        return load(*args,**kwargs)
    sd.load_clip=load_prefix
