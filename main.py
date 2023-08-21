# CMPT353 Summer 2023 Group Project
# Setu Patel (setup), Jenna Liebe (jla731), Luna Sang (dsa133)

# HOTEL REVIEWS AND RECOMMENDATIONS

import sys
import pandas as pd

sys.path.append('src')
import src.analyze_data as ad
import src.machine_learning as ml

# function to gracefully handle exit
def end_program():
    print('\nThank you for using the program!')
    quit()

def main(english_data):    
    print('\n*----------US HOTELS - Recommendations and Ratings----------*')
    print('         NLP-based model used for analyzing reviews\n')
    
    while True:               
        print('*-------------------------MAIN MENU-------------------------*')
        
        # user choice of city
        print('1. New York City')
        print('2. Chicago')
        print('3. San Francisco')
        print('4. San Diego')
        print('5. Los Angeles')
        print('6. Philadelphia')
        print('7. Dallas')
        print('8. Indianapolis')
        print('9. Boston')
        print('10. Washington DC')

        print('\nEnter "q" to exit.')

        city_choice = input('\nPlease select a city from those listed above (1-10), or exit the program (q): ')
        
        # quit the program if chosen
        if city_choice == 'q':
            end_program()
        
        # check user input (if it wasn't "q")
        try:
            city_choice = int(city_choice)
            print('\n')
        except ValueError:
            print('Invalid choice. Please enter a valid number from the list.\n')
            continue
        
        if city_choice not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print('Invalid choice. Please enter a valid number from the list.\n')
            continue
        else:
            # if the input is okay, convert the city_choice to a string for later use
            city_text = ad.city_choice_to_text(city_choice)
        
        
        while True:
            # user choice of function
            print('*-------------------------CITY MENU-------------------------*')
            print(f'1. Top 5 Hotels in {city_text} by User Reviews')
            print(f'2. Top 10 Hotels in {city_text} by Category')
            print(f'3. Browse Hotels in {city_text}')
            print('4. Choose New City')

            print('\nEnter "q" to exit.')
            
            function_choice = input('\nPlease select an option from those listed above: ')

            # quit the program if chosen
            if function_choice == 'q':
                end_program()

            # check user input (if it wasn't "q")       
            try:
                function_choice = int(function_choice)
                print('\n')
            except ValueError:
                print('Invalid choice. Please enter a valid number from the list.\n')
                continue

            #---NB model to classify positive/negative reviews, for determining top 5 hotels in the city (based on count of positive reviews)---#
            if function_choice == 1:
                # create the training and prediction data for the NB model
                print(f'\nCompiling hundreds of user reviews for hotels in {city_text}...')

                training_data, prediction_data = ad.split_by_city(english_data, city_text)               
                training_data.to_json('./filtered_data/training_data.json', orient = 'records')
                prediction_data.to_json('./filtered_data/prediction_data.json', orient = 'records')
                
                print('Successfully generated data, now analyzing results. Please hold...')
                
                # PASS DATA TO MACHINE LEARNING SCRIPT          
                model_result = ml.naive_bayes()
                
                # process the model output and print the results
                processed_model_result = ad.process_model_data(model_result)

                output_file = f'./output/top_five_{city_text}.txt'

                print(f'\nThe top 5 hotels in {city_text} based on user reviews are:')
                data_to_print = processed_model_result.drop_duplicates(subset=['name']).head(5)
                ad.print_top_results(data_to_print, output_file)
                print(f'\nThese results have been printed to output/{output_file}')
                               
                ad.create_wordcloud(english_data, data_to_print['name'])

                print('\n\n')
                continue
            

            #---Ranking the hotels in the city by review category, using NB model to impute missing scores based on positive/negative classification---#
            elif function_choice == 2:
                print('\n*-----------------Hotel Rankings by Category----------------*')
                print('1. Service\n2. Cleanliness\n3. Location\n4. Value\n5. Rooms')
                
                # checking user input again
                try:
                    cat_choice = int(input('\nPlease choose one of the above categories to compare hotels by: '))
                    if cat_choice not in [1, 2, 3, 4, 5]:
                        print('Invalid choice. Please enter a valid number from the list.')
                        continue
                except ValueError:
                    print('Invalid choice. Please enter a valid number from the list.')
                    continue

                # if user input okay, convert the choice to a category string for later use 
                cat_text = ad.cat_choice_to_text(cat_choice)
                
                # create the training and prediction data for the NB model
                print(f'\nCompiling hundreds of user reviews for hotels in {city_text}...')

                training_data, prediction_data = ad.split_by_city(english_data, city_text)               
                training_data.to_json('./filtered_data/training_data.json', orient = 'records')
                prediction_data.to_json('./filtered_data/prediction_data.json', orient = 'records')
                
                print('Successfully generated data, now analyzing results. Please hold...')
                
                # PASS DATA TO MACHINE LEARNING SCRIPT          
                model_result = ml.naive_bayes()

                # process the model output and print the results
                processed_model_result = ad.process_model_data(model_result)
                result = ad.category_rating(processed_model_result, cat_text)
                
                output_file = f'./output/top_ten_{cat_text.title()}_{city_text}.txt'

                print(f'\nThe top 10 hotels in {city_text} based on {cat_text} are:')
                ad.print_top_cat_results(result, cat_text, output_file)
                print(f'\nThese results have been printed to output/{output_file}')
                print('\n\n')
            

            #---Printing all the hotels in the city, so the user can view them independently---#
            elif function_choice == 3:
                print('\n\n*------------------------Hotel Viewer-----------------------*')
                selected_hotel = ad.hotel_list(english_data, city_text)
                
                # once the user chooses a hotel, print its output and wordcloud
                if selected_hotel:
                    ad.print_single_hotel(english_data, selected_hotel)
                    selected_hotel = [selected_hotel]
                    ad.create_wordcloud(english_data, selected_hotel)
                print('\n\n')
            
            #---Move back to the main menu to select a new city---#
            elif function_choice == 4:
                break
            
            #---Retry the user input (backup check)---#
            else:
                print('Invalid choice. Please enter a valid number from the list.\n')


if __name__ == '__main__': 
    english_data = pd.read_csv('filtered_data/all_cities_cleaned_english_reviews.csv.gz')
    main(english_data)