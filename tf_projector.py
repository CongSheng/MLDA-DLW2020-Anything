import io
def export_to_tf_projector(model, name):
    out_v = io.open('{}_vecs.tsv'.format(name), 'w', encoding='utf-8')
    out_m = io.open('{}_meta.tsv'.format(name), 'w', encoding='utf-8')

    for word in model.words:
        vec = model.get_word_vector(word)
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")

    out_v.close()
    out_m.close()
