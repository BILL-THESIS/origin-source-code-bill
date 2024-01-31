#!/bin/bash

# Set the paths
INPUT_DATA="../../Sonar/seatunnel_all_information.parquet"
OUTPUT_PATH="../../output/cluster"
COMBINE_PATH="../../models/Kmeans/combia2-copy"
NUM_CPUS=5

# Function to read Parquet files and combine them
read_parquet() {
    df_list=()
    for parquet_file in "$COMBINE_PATH"/*.parquet; do
        df=$(pd.read_parquet "$parquet_file")
        df_list+=("$df")
        echo "DF List :: $df_list"
    done
}

# Function to scale data
scale_data() {
    local scaler_list=()
    for data_scaler in "$@"; do
        scaled=$(scaler.fit_transform "$data_scaler")
        scaled_df=$(pd.DataFrame "$scaled", columns="$data_scaler.columns")
        scaler_list+=("$scaled_df")
        unset scaled_df scaled
    done
    echo "${scaler_list[@]}"
}


# Function to perform KMeans clustering
kmeans_cluster() {
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Start to normalize cluster at: $start_time"

    local list_label=()
    local i
    local n_clusters
    for data_scaler in $(scale_data "$@"); do
        for i in "${df_list[@]}"; do
            for n_clusters in {2..4}; do
                kmeans=(KMeans -n_clusters "$n_clusters" -n_init 10)
                cluster_labels=("$("${kmeans[@]}" fit_predict "$data_scaler")")
                df_cluster_labels=$(pd.DataFrame "$cluster_labels")
                clusters=$(silhouette_score "$data_scaler" "${kmeans[@]}" labels | awk '{printf "%.4f", $1}')
                list_label+=("$i" "$df_cluster_labels" "$clusters" "$n_clusters")
            done
        done
    done

    for ((idx=0; idx<${#list_label[@]}; idx+=4)); do
        i=${list_label[idx]}
        cluster_labels=${list_label[idx+1]}
        clusters=${list_label[idx+2]}
        n_clusters=${list_label[idx+3]}

        df_concat_col=$(pd.DataFrame "$i")
        df_concat_col['cluster_labels']="$cluster_labels"
        df_concat_col['score']="$clusters"
        df_concat_col['clusters']="$n_clusters"

        output_path="$OUTPUT_PATH$n_clusters"
        merged_df_original=$(pd.concat "$df_original['total_time']" "$df_concat_col" axis=1 | reindex "$df_concat_col.index" | awk '!seen[$0]++')
        echo "merged_df_original $merged_df_original"
        echo "$merged_df_original" | to_parquet "$output_path/${i.columns.to_list}_${n_clusters}.parquet"
    done

    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    result_time=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
    result_time_gmt=$(date -u -d @$result_time +"%H:%M:%S")
    echo "Total time: $result_time_gmt"
}

# Limit to just 120000 rows
list_df_list=("${df_list[@]:0:120000}")

# Parsed data is data files combination part from list df
parsed_description_split=("${list_df_list[@]:0:2}" "${list_df_list[@]:2:2}" "${list_df_list[@]:4:2}" "${list_df_list[@]:6:2}" "${list_df_list[@]:8:2}")

echo "Processing files using $NUM_CPUS CPU cores"
echo "Number of splits: ${#parsed_description_split[@]}"

# Function to get DataFrame cluster
get_df_cluster() {
    start=$(date +"%Y-%m-%d %H:%M:%S")
    get_cluster=$(kmeans_cluster "$@")
    end=$(date +"%Y-%m-%d %H:%M:%S")
#    get_time=$(( $(date -d "$end" +%s) - $(date -d "$start" +%s) ))
    echo "Total time get cluster: $get_time"
    echo "$get_cluster"
}

# Execute the clustering process using multiprocessing
parsed_description_split=("${parsed_description_split[@]}")
get_df_cluster "${parsed_description_split[@]}"
