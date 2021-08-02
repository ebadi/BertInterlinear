from fitbert import FitBert


# currently supported models: bert-large-uncased and distilbert-base-uncased
# this takes a while and loads a whole big BERT into memory
fb = FitBert()

"""
masked_string = "Why ***mask***, you're looking ***mask*** today!"
options = ['buff', 'handsome', 'strong']

ranked_options = fb.rank(masked_string, options=options)
print(ranked_options)
# >>> ['handsome', 'strong', 'buff']
# or
filled_in = fb.fitb(masked_string, options=options)
# >>> "Why Bert, you're looking handsome today!"

print(filled_in)
"""
masked_string = "Hello  ***mask*** test ***mask*** today!"


options1 = ['looking', 'catching', 'master', 'handsome', ]
options2 = ['rank', 'book', 'strong']
filled_in = fb.rank_multi(masked_string, options=options1)
print("rank_multi", filled_in)


filled_in1 = fb.new_rank_multi(masked_string, words=options1)
print("new_rank_multi", filled_in1)
