# Provides the analysis functions that are called in main.py.

import pandas as pd
import src.WordCloud as wc
import re

#------------------------------------------------------------------------------------------#
# Converts the city selection (an int) into text, for column specification and output
def city_choice_to_text(city_choice):
    if city_choice == 1:
        text = 'New York City'
    elif city_choice == 2:
        text = 'Chicago'
    elif city_choice == 3:
        text = 'San Francisco'
    elif city_choice == 4:
        text = 'San Diego'
    elif city_choice == 5:
        text = 'Los Angeles'
    elif city_choice == 6:
        text = 'Philadelphia'
    elif city_choice == 7:
        text = 'Dallas'
    elif city_choice == 8:
        text = 'Indianapolis'
    elif city_choice == 9:
        text = 'Boston'
    elif city_choice == 10:
        text = 'Washington DC'
    return text

#------------------------------------------------------------------------------------------#
# Converts the category selection (an int) into text, for column specification and output
def cat_choice_to_text(cat_choice):
    if cat_choice == 1:
        text = 'service'
    elif cat_choice == 2:
        text = 'cleanliness'
    elif cat_choice == 3:
        text = 'location'
    elif cat_choice == 4:
        text = 'value'
    elif cat_choice == 5:
        text = 'rooms'
    
    return text

#------------------------------------------------------------------------------------------#
# Splits data by specified city
def split_by_city(data, city):
    model_data = data[['title', 'text', 'comment_id', 'review', 'locality']]
    
    model_data = model_data.copy()
    model_data.loc[:, 'review_text'] = model_data['title'] + ' ' + model_data['text']
    model_data = model_data.drop(['title', 'text'], axis = 1)
    
    training_data = model_data[model_data['locality'] != city]
    validation_data = model_data[model_data['locality'] == city]
    
    training_data = training_data.drop(['locality'], axis = 1)
    validation_data = validation_data.drop(['locality'], axis = 1)
    
    return training_data, validation_data

#------------------------------------------------------------------------------------------#
# Sorts hotel data by the weighted average score of the specified category
def category_rating(data, cat_text): 
    review_count = data['name'].value_counts()
    
    data = data.copy()
    data.loc[:, 'review_count'] = data['name'].map(review_count)
    
    data.loc[:, cat_text] = data[cat_text].fillna(data['predicted_review_label'].map({'positive': 4, 'negative': 2}))
    
    overall_avg = data[cat_text].mean()

    # Calculate the weighted average using a smoothing factor - gives more weight to hotels with higher # of reviews, but still consider avg score as baseline
    smoothing_factor = 10
    data.loc[:, 'weighted_avg'] = (data['review_count'] * data[cat_text] + smoothing_factor * overall_avg) / (data['review_count'] + smoothing_factor)
    
    # sort the data by the weighted_avg and positive_counts (# of positive comments), then drop duplicates
    sorted_data = data.sort_values(by = ['weighted_avg', 'positive_counts'], ascending = False)
    sorted_data = sorted_data.drop_duplicates(subset = 'name', keep = 'first')

    # keep just the top ten hotels
    top_results = sorted_data[['name', 'weighted_avg', 'review_count', 'positive_counts']].head(10)
    
    return top_results


#------------------------------------------------------------------------------------------#
# Helper function for hotel_list
def print_hotels(data, start_index, items_per_page = 20):
    end_index = min(start_index + items_per_page, len(data))
    for i in range(start_index, end_index):
        print(f'{i + 1}. {data.iloc[i]}')
    if end_index >= len(data):
        print('\nThere are no more hotels to display for this city.')


#------------------------------------------------------------------------------------------#
# Function that prints out the list of hotels (by city) for the specific hotel data functionality
def hotel_list(data, city):
    current_page = 0
    hotels_to_print = data.loc[data['locality'] == city, 'name']
    hotels_to_print = hotels_to_print.drop_duplicates()
    
    while True:
        print_hotels(hotels_to_print, current_page * 20)
        print("\nOptions:")
        print("N: Next 20 hotels")
        print("P: Previous 20 hotels")
        print("Q: Quit")
        print("Or select the number of the hotel")
        
        user_input = input("Please select an option from those listed above: ").strip().lower()
        print('\n')

        if user_input == 'n':
            current_page += 1
        elif user_input == 'p':
            if current_page > 0:
                current_page -= 1
        elif user_input.isdigit():
            hotel_index = int(user_input) - 1
            if hotel_index >= 0 and hotel_index < len(hotels_to_print):
                selected_hotel = hotels_to_print.iloc[hotel_index]
                return selected_hotel
            else:
                print("Invalid hotel number.")
        elif user_input == 'q':
            return False
        else:
            print("Invalid input. Please try again.")


#------------------------------------------------------------------------------------------#
# Function to process the output from the NB model (count up total positives/negatives)
def process_model_data(model_data):
    # Total reviews for each hotel
    review_count = model_data['name'].value_counts()
    model_data = model_data.copy()
    model_data.loc[:, 'review_count'] = model_data['name'].map(review_count)

    # Total positive reviews for each hotel
    positive_counts = model_data.groupby('name')['predicted_review_label'].apply(lambda x: sum(x == 'positive'))
    model_data = model_data.merge(positive_counts.rename('positive_counts'), on = 'name')    
    
    model_data.sort_values(by=['positive_counts'], ascending = False, inplace = True)

    return model_data


#------------------------------------------------------------------------------------------#
# Nicely formats the printing of multiple hotels for the category scoring
def print_top_cat_results(data, cat_text, output_file):   
    max_name_length = data['name'].str.len().max()
    
    avg_cat_header = f"Avg {cat_text.title()} Score"
    review_count_header = 'Total Count'
    positive_count_header = 'Positive Reviews'
    
    rank_width = 5
    avg_score_width = 22
    review_count_width = 12
    positive_counts_width = 16
    
    header = f"{'Rank':<{rank_width}} {'Hotel Name':<{max_name_length}} {avg_cat_header:<{avg_score_width}} {review_count_header:<{review_count_width}} {positive_count_header:<{positive_counts_width}}"
    separator = '-' * (max_name_length + 50) + '\n'
    table_line = '\n'

    # print to command line
    print(header.strip())
    print(separator.strip())
    for rank, row in enumerate(data.iterrows(), 1):
        hotel_name = row[1]['name']
        weighted_avg = row[1]['weighted_avg']
        review_count = row[1]['review_count']
        positive_counts = row[1]['positive_counts']
        print(f"{rank:<{rank_width}} {hotel_name:<{max_name_length}} {weighted_avg:<{avg_score_width}.2f} {review_count:<{review_count_width}} {positive_counts:<{positive_counts_width}}")

    # print to output file
    with open(output_file, 'w') as f:
        f.write(header)
        f.write(table_line)
        f.write(separator)
        for rank, row in enumerate(data.iterrows(), 1):
            hotel_name = row[1]['name']
            weighted_avg = row[1]['weighted_avg']
            review_count = row[1]['review_count']
            positive_counts = row[1]['positive_counts']
            f.write(f"{rank:<{rank_width}} {hotel_name:<{max_name_length}} {weighted_avg:<{avg_score_width}.2f} {review_count:<{review_count_width}} {positive_counts:<{positive_counts_width}}")
            f.write(table_line)


#------------------------------------------------------------------------------------------#
# Nicely formats the printing of multiple hotels for the top 5 model scoring
def print_top_results(data, output_file):
    max_name_length = data['name'].str.len().max()
    
    review_count_header = 'Total Reviews'
    positive_count_header = 'Positive Reviews'
    
    rank_width = 5
    review_count_width = 12
    positive_counts_width = 16
    
    header = f"{'Rank':<{rank_width}} {'Hotel Name':<{max_name_length}} {review_count_header:<{review_count_width}} {positive_count_header:<{positive_counts_width}}"
    separator = '-' * (max_name_length + 50) + '\n'
    table_line = '\n'
    
    # print to command line
    print(header.strip())
    print(separator.strip())
    for rank, (index, row) in enumerate(data.iterrows(), 1):
        hotel_name = row['name']
        review_count = row['review_count']
        positive_counts = row['positive_counts']
        print(f"{rank:<{rank_width}} {hotel_name:<{max_name_length}} {review_count:<{review_count_width}} {positive_counts:<{positive_counts_width}}")

    # print to output file
    with open(output_file, 'w') as f:
        f.write(header)
        f.write(table_line)
        f.write(separator)
        for rank, (index, row) in enumerate(data.iterrows(), 1):
            hotel_name = row['name']
            review_count = row['review_count']
            positive_counts = row['positive_counts']
            f.write(f"{rank:<{rank_width}} {hotel_name:<{max_name_length}} {review_count:<{review_count_width}} {positive_counts:<{positive_counts_width}}")
            f.write(table_line)


#------------------------------------------------------------------------------------------#
# Printing format + word cloud generation for the specific hotel data functionality
def print_single_hotel(data, selected_hotel):
    hotel_data = data.loc[data['name'] == selected_hotel]
    hotel_info = hotel_data.iloc[0]
    hotel_url = hotel_info['url']
    hotel_address = f"{hotel_info['street-address']}, {hotel_info['locality']}, {hotel_info['region']}"
    hotel_class = hotel_info['hotel_class']
    hotel_reviews = hotel_data.shape[0]

    if not hotel_class:
        hotel_class = 'Not Given'

    box_char = "-"
    box_width = 50

    # Print the box header with the hotel name as title
    print(f'\n{box_char * box_width}')
    print(f'{box_char} {selected_hotel.center(box_width - 4)} {box_char}')
    print(f'{box_char * box_width}')
    
    print(f'Hotel Class: {hotel_class} stars')
    print(f'Hotel Address: {hotel_address}')
    print(f'TripAdvisor Link: {hotel_url}')
    print(f'Number of Reviews: {hotel_reviews}')

    # Print the hotel details to hotel-details.txt, in the output/ folder
    out_file_path = f'./output/hotel-details-{selected_hotel}.txt'

    with open(out_file_path, "w") as file:
        file.write(f'\n{box_char * box_width}\n')
        file.write(f'{box_char} {selected_hotel.center(box_width - 4)} {box_char}\n')
        file.write(f'{box_char * box_width}\n\n')
        
        file.write(f'Hotel Class: {hotel_class} stars\n')
        file.write(f'Hotel Address: {hotel_address}\n')
        file.write(f'TripAdvisor Link: {hotel_url}\n')
        file.write(f'Number of Reviews: {hotel_reviews}')
    

    print(f'The hotel details have been printed to the output folder as hotel-details-{selected_hotel}.txt.')


#------------------------------------------------------------------------------------------#
# Generates a wordcloud for the input hotel
def create_wordcloud(data, hotels):
    print('\n**NOTE: to return to the menu, please close the word cloud(s).**')
    
    # default setting - display the first wordcloud
    wc_choice = 0

    for rank, hotel_name in enumerate(hotels, 1):        
        # Filter the data for the current hotel
        hotel_data = data.loc[data['name'] == hotel_name]   
        hotel_full_name = hotel_data['name'].iloc[0]
        
        # Using the title for wordcloud, because it contains fewer irrelevant words (such as 'husband' or 'stayed', generally)
        hotel_comments = hotel_data['title']

        cleaned_name = re.sub(r'\W+', '_', hotel_full_name)
        cleaned_name = cleaned_name.strip('_')
        file_name = f"./output/wordcloud-{cleaned_name}.png"

        wc.generate_wordcloud(hotel_comments, display_wc=wc_choice, output_file=file_name, mask_path=None, stopwords=None, max_words=1000, background_color="white",  width=800, height=800)
        print(f'This wordcloud has been saved to your machine as wordcloud-{cleaned_name}.png, in the output folder.')

        if rank == 1 and len(hotels) > 1:
            # Ask the user on the first wordcloud if they want to see the rest (for top 5 hotels) - if they enter 1 or anything else != 0, only the first one will be displayed
            wc_choice = input("\nIf you'd like to see the other wordclouds, enter 0. Otherwise enter 1 - they will not be displayed (but still saved): ")
            
            try:
                wc_choice = int(wc_choice)
                print('\n')
            except ValueError:
                print('Invalid choice. The remaining wordclouds will not be displayed.\n')
                wc_choice = 1
                continue

            if wc_choice not in [0, 1]:
                print('Invalid choice. The remaining wordclouds will not be displayed.\n')
                wc_choice = 1
                continue
                

    