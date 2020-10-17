import numpy as np
import pandas as pd
import os

print("Imports are ready")

########################################################
### combine shipment_id, phone_id, user_id, order_id ###
########################################################

# First, load the files containing information about shipments
shipments1 = pd.read_csv("./ngwl-predict-customer-churn/shipments/shipments2020-03-01.csv")
shipments2 = pd.read_csv("./ngwl-predict-customer-churn/shipments/shipments2020-01-01.csv")
shipments3 = pd.read_csv("./ngwl-predict-customer-churn/shipments/shipments2020-04-30.csv")
shipments4 = pd.read_csv("./ngwl-predict-customer-churn/shipments/shipments2020-06-29.csv")

# Put all shipments into one table
shipments = pd.concat([shipments1, shipments2, shipments3, shipments4])

# Read addresses and fix the column names
addresses = pd.read_csv("./ngwl-predict-customer-churn/misc/addresses.csv")
addresses.columns = ["ship_address_id", "phone_id"]

# Now create the mapping through shipment_address_id with the addresses to receive phone_id
shipments_and_addresses = pd.merge(addresses, shipments, on = "ship_address_id")

# We will take the phone id, user id, shipment id, order id, order state from here
final_table = pd.DataFrame()
final_table["phone_id"] = shipments_and_addresses.phone_id
final_table["user_id"] = shipments_and_addresses.user_id
final_table["shipment_id"] = shipments_and_addresses.shipment_id
final_table["order_id"] = shipments_and_addresses.order_id
final_table["order_state"] = shipments_and_addresses["s.order_state"]

#########################################################
### extract data from the line_items about categories ###
#########################################################

def find_month_categories_first(directory):
    """
    This function creates a pandas DataFrame for each month, whereas
    for each shipment_id only the firs item that was put inside the
    cart is saved. If several items were put at the same time, only
    one is saved for the corresponding shipment_id. Also only the
    orders, that was not cancelled are saved.
    
    :param directory: directory for a month to read the files from
    :returns: a pandas DataFrame where for each shipment_id there is
        one row with corresponding information.
    """
    all_month_categories = []

    for filename in os.listdir(directory):

        current_file = pd.read_csv(os.path.join(directory, filename))
        current_table = pd.DataFrame(columns = current_file.columns)

        # take only not cancelled orders
        current_file = current_file.loc[current_file['cancelled'] == 0]
        
        all_s_ids = current_file.shipment_id.unique()
        count = 1 # meant for visualization of the progress
        
        for s in all_s_ids:
            shipment_table = current_file.loc[current_file['shipment_id'] == s]
            # find the earliest date
            s_row = shipment_table.loc[shipment_table['created_at'] == shipment_table.created_at.min()]
            current_table = current_table.append(s_row.iloc[0])
            print("appended", count, "/", len(all_s_ids), filename)
            count += 1

        all_month_categories.append(current_table)

    month_categories = pd.DataFrame(pd.concat(all_month_categories, axis = 0, ignore_index = True))
    
    return month_categories

# Now load the information using the above function
jan_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items01")
feb_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items02")
mar_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items03")
apr_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items04")
mai_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items05")
jun_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items06")
jul_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items07")
aug_categories = find_month_categories_first("./ngwl-predict-customer-churn/line_items08")


########################################################
### combine all extracted information into one table ###
########################################################

def create_final_month_report(categories):
    """
    This function combines the information about categories wiht the phone_id, user_id, and
    creates a final pandas DataFrame for each month, that can be used for the further work.
    :param categories: extracted categories in the previous step
    :returns: a pandas DataFrame for each month
    """
    month = pd.merge(categories, final_table, on = "shipment_id")
    #rearrange the columns
    cols = month.columns.tolist() #[....., 'phone_id', 'user_id', 'order_id', 'order_state']
    cols = cols[-4:] + cols[:-4] #['phone_id', 'user_id', 'order_id', 'order_state', ......]
    month = month[cols]
    
    return month

# Now use the created function to create tables
jan = create_final_month_report(jan_categories)
feb = create_final_month_report(feb_categories)
mar = create_final_month_report(mar_categories)
apr = create_final_month_report(apr_categories)
mai = create_final_month_report(mai_categories)
jun = create_final_month_report(jun_categories)
jul = create_final_month_report(jul_categories)
aug = create_final_month_report(aug_categories)

# Save generated data to .csv files
jan.to_csv("./jan.csv", index=False)
feb.to_csv("./feb.csv", index=False)
mar.to_csv("./mar.csv", index=False)
apr.to_csv("./apr.csv", index=False)
mai.to_csv("./mai.csv", index=False)
jun.to_csv("./jun.csv", index=False)
jul.to_csv("./jul.csv", index=False)
aug.to_csv("./aug.csv", index=False)
