import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
all_df = pd.read_csv("all_data.csv")
orders_review_df = pd.read_csv("orders_review.csv")
orders_item_df = pd.read_csv("orders_item.csv")

# Convert timestamps to datetime
all_df['review_creation_date'] = pd.to_datetime(all_df['review_creation_date'], errors='coerce', infer_datetime_format=True)
all_df['order_purchase_timestamp'] = pd.to_datetime(all_df['order_purchase_timestamp'], errors='coerce', infer_datetime_format=True)

# Define the minimum and maximum date based on your data
min_date = all_df['order_purchase_timestamp'].min()
max_date = all_df['order_purchase_timestamp'].max()

# Sidebar input for date range
with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Filter the data based on the selected date range
filtered_df = all_df[(all_df['order_purchase_timestamp'] >= pd.to_datetime(start_date)) & 
                     (all_df['order_purchase_timestamp'] <= pd.to_datetime(end_date))]

# Days between order and review
days_to_review = (filtered_df['review_creation_date'] - filtered_df['order_purchase_timestamp']).dt.days
filtered_df["days_to_review"] = days_to_review

# Create tabs for each question and visualization
tab1, tab2, tab3, tab4 = st.tabs(["Waktu Pemesanan vs Review", "Tingkat Pengiriman vs Rating", "Pesanan & Pendapatan", "Pengaruh Diskon"])

# Tab 1: Pola Hubungan Waktu Pemesanan dengan Review Pelanggan
with tab1:
    st.subheader('Distribution of Days Between Order and Review')

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(filtered_df['days_to_review'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Days Between Order and Review')
    ax.set_xlabel('Days between Order and Review')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader('Scatter Plot of Order Time vs Review Time')

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.scatterplot(x=filtered_df['order_purchase_timestamp'], y=filtered_df['review_creation_date'], ax=ax)
    ax.set_title('Scatter Plot of Order Time vs Review Time')
    ax.set_xlabel('Order Time')
    ax.set_ylabel('Review Time')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader('Boxplot of Days Between Order and Review')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(filtered_df['days_to_review'], ax=ax)
    ax.set_title('Boxplot of Days Between Order and Review')
    ax.set_xlabel('Days to Review')
    st.pyplot(fig)

# Tab 2: Hubungan Pengiriman Tepat Waktu dengan Rating Produk
with tab2:
    # Proporsi pengiriman tepat waktu vs terlambat
    orders_review_df['delivery_on_time'] = orders_review_df['order_delivered_customer_date'] <= orders_review_df['order_estimated_delivery_date']
    delivery_status = orders_review_df['delivery_on_time'].value_counts(normalize=True) * 100

    st.subheader('Proportion of On-time vs Late Deliveries')

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=delivery_status.index, y=delivery_status.values, ax=ax)
    ax.set_title('Proportion of On-time vs Late Deliveries')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Late', 'On-Time'])
    ax.set_ylabel('Percentage')
    st.pyplot(fig)

    # Boxplot hubungan antara pengiriman tepat waktu dan review score
    st.subheader('Review Score Distribution for On-time vs Late Deliveries')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='delivery_on_time', y='review_score', data=orders_review_df, ax=ax)
    ax.set_title('Review Score Distribution for On-time vs Late Deliveries')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Late', 'On-Time'])
    ax.set_ylabel('Review Score')
    st.pyplot(fig)

# Tab 3: Hubungan Jumlah Pesanan dengan Total Pendapatan per Bulan
with tab3:
    # Resampling dan agregasi data untuk rentang waktu yang dipilih
    monthly_orders_df = filtered_df.resample(rule='M', on='order_purchase_timestamp').agg({
        "order_id": "nunique",  # Menghitung jumlah pesanan unik
        "total_price": "sum"    # Menghitung total pendapatan
    })

    # Mengubah format indeks menjadi nama bulan
    monthly_orders_df.index = monthly_orders_df.index.strftime('%B')

    # Mereset index dan mengganti nama kolom
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "order_id": "order_count",
        "total_price": "revenue"
    }, inplace=True)

    # Visualisasi jumlah pesanan per bulan
    st.subheader('Number of Orders per Month (Selected Range)')

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(monthly_orders_df["order_purchase_timestamp"], monthly_orders_df["order_count"], marker='o', linewidth=2, color="#72BCD4")
    ax.set_title("Number of Orders per Month", fontsize=20)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Number of Orders", fontsize=14)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    ax.grid(True)
    st.pyplot(fig)

    # Visualisasi total pendapatan per bulan
    st.subheader('Total Revenue per Month (Selected Range)')

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(monthly_orders_df["order_purchase_timestamp"], monthly_orders_df["revenue"], marker='o', linewidth=2, color="#FF6F61")
    ax.set_title("Total Revenue per Month", fontsize=20)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Revenue", fontsize=14)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    ax.grid(True)
    st.pyplot(fig)

# Tab 4: Pengaruh Diskon atau Promosi terhadap Volume Penjualan
with tab4:
    # Menambahkan kolom untuk 'diskon' (proxy dengan freight_value atau selisih harga)
    all_df['diskon'] = all_df['freight_value'] < all_df['freight_value'].mean()

    # Menghitung volume penjualan berdasarkan diskon
    sales_by_discount = all_df.groupby('diskon')['order_id'].count().reset_index()

    # Visualisasi
    st.subheader('Volume Penjualan Berdasarkan Diskon')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='diskon', y='order_id', data=sales_by_discount, ax=ax)
    ax.set_title('Volume Penjualan Berdasarkan Diskon')
    ax.set_xlabel('Diskon')
    ax.set_ylabel('Jumlah Pesanan')
    st.pyplot(fig)