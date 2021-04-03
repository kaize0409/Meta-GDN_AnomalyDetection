def modelArch(in_feature, out_feature):

    config = [
        ('linear', [512, in_feature]),
        # ('relu', [True]),
        ('linear', [out_feature, 512])
    ]

    return config