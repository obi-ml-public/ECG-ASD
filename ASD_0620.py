SHAPE=(2500, 12, 1)

def multi_conv2D(x, num_kernel, activation="relu"):
    kreg = None
    skip = Conv2D(num_kernel*3, 1, activation=None, padding="same")(x)
    skip = BatchNormalization()(skip)
    a = Conv2D(int(num_kernel), 1, activation=activation, padding="same", kernel_regularizer=kreg)(x)
    a = BatchNormalization()(a)
    a = Conv2D(num_kernel, 3, activation=activation, padding="same", kernel_regularizer=kreg)(a)
    a = BatchNormalization()(a)
    b = Conv2D(int(num_kernel), 1, activation=activation, padding="same", kernel_regularizer=kreg)(x)
    b = BatchNormalization()(b)
    b = Conv2D(int(num_kernel), 3, activation=activation, padding="same", kernel_regularizer=kreg)(b)
    b = BatchNormalization()(b)
    b = Conv2D(num_kernel, 3, activation=activation, padding="same", kernel_regularizer=kreg)(b)
    b = BatchNormalization()(b)

    c = Conv2D(int(num_kernel), 1, activation=activation,padding="same", kernel_regularizer=kreg)(x)
    c = BatchNormalization()(c)
    res = Concatenate()([a,b,c])
    res = BatchNormalization()(res)
    res = Add()([res,skip])
    return res
def get_models_2D():
    input1 = Input(SHAPE)
    initial_kernel_num = 64
    x = input1
    x = Conv2D(initial_kernel_num, (7,3),strides=(2,1), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = multi_conv2D(x, initial_kernel_num)
    x = multi_conv2D(x, initial_kernel_num)
    x = MaxPooling2D(pool_size=(3,1))(x)
        
    x = multi_conv2D(x, int(initial_kernel_num*1.5))
    x = multi_conv2D(x, int(initial_kernel_num*1.5))
    x = MaxPooling2D(pool_size=(3,1))(x)
  
    x = multi_conv2D(x, int(initial_kernel_num*2))
    x = multi_conv2D(x, int(initial_kernel_num*2))
    x = MaxPooling2D(pool_size=(2,1))(x)
  
    x = multi_conv2D(x, initial_kernel_num*3)
    x = multi_conv2D(x, initial_kernel_num*3)
    x = multi_conv2D(x, initial_kernel_num*4)
    x = MaxPooling2D(pool_size=(2,1))(x)
    
    x = multi_conv2D(x, initial_kernel_num*5)
    x = multi_conv2D(x, initial_kernel_num*6)
    x = multi_conv2D(x, initial_kernel_num*7)
    x = MaxPooling2D(pool_size=(2,1))(x)
    
    x = multi_conv2D(x, initial_kernel_num*8)
    x = multi_conv2D(x, initial_kernel_num*8)
    x = multi_conv2D(x, initial_kernel_num*8)
    x = MaxPooling2D(pool_size=(2,1))(x)
   
    x = multi_conv2D(x, initial_kernel_num*12)
    x = multi_conv2D(x, initial_kernel_num*14)
    x = multi_conv2D(x, initial_kernel_num*16)
    x = GlobalAveragePooling2D()(x)
    
    a1 = Dense(1,activation='sigmoid')(x)

    model = Model(inputs=input1, outputs=a1)
    return model

model = get_models_2D()