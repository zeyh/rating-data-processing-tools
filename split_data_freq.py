from func import take_arg,extract_denser

def main():
    filepath = take_arg(1)
    if(filepath == []):
        filepath = ["dataset/movielen-20m","dataset/sub/extracted_"]
    print(filepath)
    for i in range(5):
        print("now split subset #",i)
        # filepath[0] = 
        extract_denser(filepath[0], 50+50*i, i)

    print("fin")

if __name__ == '__main__':
    main()