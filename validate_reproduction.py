"""Numerical checks against functions read directly from the released TF1 code."""
import ast
import json
import math
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import tensorflow as tensorflow
import torch
import reproduction as repro


def validate():
    tf = tensorflow.compat.v1
    source_path = Path('reference/Genre-Extraction/softmaxautoencoder.py')
    tree = ast.parse(source_path.read_text())
    function_names = ['add_parameters', 'encoder', 'decoder', 'softmaxclassifier']
    code = '\n\n'.join(ast.get_source_segment(source_path.read_text(), node)
                         for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name in function_names)
    code = code.replace('tf.contrib.layers.xavier_initializer()', 'tf.glorot_uniform_initializer()')
    graph = tf.Graph()
    report = {'source_commit': repro.SOURCE_COMMIT, 'checks': {}}
    with graph.as_default():
        namespace = {'tf': tf, 'input_dim': 500, 'num_classes': 4}
        exec(compile(code, str(source_path), 'exec'), namespace)
        x = tf.placeholder(tf.float32, (None, 500))
        y = tf.placeholder(tf.float32, (None, 4))
        w = namespace['add_parameters']()
        z = namespace['encoder'](x, w, 1.0)
        d = namespace['decoder'](z, w, 1.0)
        logits = namespace['softmaxclassifier'](z, w, 1.0)
        mse = tf.reduce_mean(tf.square(d-x))
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
        loss = mse + .1*ce
        train = tf.train.AdamOptimizer(1e-4).minimize(loss)
        gradients = tf.gradients(loss, list(w.values()))
        init = tf.global_variables_initializer()
        initial_values = {k: tf.placeholder(tf.float32, v.shape) for k,v in w.items()}
        assignments = [tf.assign(w[k], initial_values[k]) for k in w]
    repro.seed_everything(42)
    model = repro.Autoencoder(joint=True, keep_probability=1.0).cpu()
    mapping = {}
    for group, name in [(model.encoder, 'encoder'), (model.decoder, 'decoder'), (model.head, 'softmax')]:
        for i, layer in enumerate(group,1):
            mapping[f'W{i}_{name}'] = (layer.weight, True)
            mapping[f'b{i}_{name}'] = (layer.bias, False)
    reader = tensorflow.train.load_checkpoint('reference/Genre-Extraction/modelWeights/softmaxautoencoder-200')
    # Original trained weights exercise nontrivial encodings, not only near-zero random logits.
    for key,(parameter, transpose) in mapping.items():
        a = reader.get_tensor(key)
        with torch.no_grad():
            parameter.copy_(torch.from_numpy(a.T if transpose else a.reshape(-1)))
    values = {key: reader.get_tensor(key) for key in w}
    rng = np.random.default_rng(57)
    batch = rng.normal(0,.08,(32,500)).astype('float32')
    labels = rng.integers(0,4,32)
    onehot = np.eye(4,dtype='float32')[labels]
    xt,yt=torch.from_numpy(batch),torch.from_numpy(labels)
    session_config = tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2,
                                    device_count={'GPU':0})
    with tf.Session(graph=graph,config=session_config) as session:
        session.run(init)
        session.run(assignments,feed_dict={initial_values[k]:values[k] for k in w})
        expected = session.run([z,d,logits,loss,gradients],{x:batch,y:onehot})
        model.eval()
        outputs=model(xt)
        actual_loss,_,_=repro.objective(outputs,xt,yt)
        actual_loss.backward()
        errors={}
        for name, actual, target in [('encoding',outputs['encoding'].detach().numpy(),expected[0]),
                                    ('reconstruction',outputs['reconstruction'].detach().numpy(),expected[1]),
                                    ('logits',outputs['logits'].detach().numpy(),expected[2])]:
            np.testing.assert_allclose(actual,target,rtol=3e-4,atol=3e-5)
            errors[name] = float(np.max(np.abs(actual-target)))
        np.testing.assert_allclose(actual_loss.item(),expected[3],rtol=1e-5,atol=1e-6)
        errors['loss']=float(abs(actual_loss.item()-expected[3]))
        gradient_error=0.
        for (key,_),target in zip(w.items(),expected[4]):
            parameter,transpose=mapping[key]
            actual=parameter.grad.detach().numpy()
            actual=actual.T if transpose else actual.reshape(1,-1)
            np.testing.assert_allclose(actual,target,rtol=2e-3,atol=5e-5)
            gradient_error=max(gradient_error,float(np.max(np.abs(actual-target))))
        errors['gradient_max_absolute_error']=gradient_error
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,foreach=False)
        # Several updates verify epsilon placement and bias corrections.
        for step in range(1,11):
            optimizer.zero_grad(set_to_none=True)
            objective,_,_=repro.objective(model(xt),xt,yt)
            objective.backward()
            optimizer.param_groups[0]['eps']=1e-8/math.sqrt(1-.999**step)
            optimizer.step()
            session.run(train,{x:batch,y:onehot})
        update_error=0.
        final_values=session.run(w)
        for key,(parameter,transpose) in mapping.items():
            actual=parameter.detach().numpy()
            actual=actual.T if transpose else actual.reshape(1,-1)
            target=final_values[key]
            np.testing.assert_allclose(actual,target,rtol=2e-3,atol=3e-5)
            update_error=max(update_error,float(np.max(np.abs(actual-target))))
        errors['ten_adam_updates_max_absolute_error']=update_error
        report['checks']['original_tf1_forward_gradient_adam']=errors

    # Native-rate librosa.load is equivalent to soundfile for these mono PCM recordings.
    import librosa
    audio_root=Path('data/original/genres')
    for genre in repro.GENRES:
        path=audio_root/genre/f'{genre}.00000.wav'
        audio,sr,_=repro.read_audio(path)
        original,sr2=librosa.load(path,sr=22050,mono=True)
        pooled=repro.pool_song(audio,sr)
        original_pooled=original[:600000].reshape(15000,40).mean(1)[:10000].reshape(20,500)
        np.testing.assert_array_equal(pooled,original_pooled)
        report['checks'][f'{genre}_pooling_max_error']=float(np.max(abs(pooled-original_pooled)))
    # Check both formulas independently with known targets.
    dummy={'reconstruction':torch.ones(2,500),'logits':torch.zeros(2,4)}
    inp=torch.zeros(2,500);label=torch.zeros(2,dtype=torch.long)
    for mode,expected in [('repository',1+.1*math.log(4)),('paper_literal',450+.1*math.log(4))]:
        actual=repro.objective(dummy,inp,label,mode)[0].item()
        np.testing.assert_allclose(actual,expected,rtol=1e-6)
        report['checks'][mode+'_formula']=True
    report['passed']=True
    Path('analysis/numerical_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    validate()
