from func import take_arg,extract_denser

def main():
    filepath = take_arg(1)
    if(filepath == []):
        filepath = ["dataset/movielen-20m","dataset/sub/extracted_"]
    print(filepath)
    for i in range(5):
        print(i)
        extract_denser(filepath[0], 50+50*i, i)

    print("fin")

main()