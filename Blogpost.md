# Generating synthetic data in a safe way with metasyn


## Introduction

Doing open, reproducible science means doing your best to openly share research data and analysis code. With these open materials, others can check and understand your research, use it to prepare their own analysis, find examples for teaching, and more. However, sometimes datasets contain sensititive or confidential information, which makes it difficult — if not impossible — share. In this case, producing and sharing a _synthetic_ version of the data might be a solution. In this post, we show how to do this in an auditable, transparent way with the software package `metasyn`.

Metasyn is a Python package that helps you to generate synthetic data, with two ideas in mind. First, it is easy to use and understand. Second, and most importantly, it is privacy-friendly.  Unlike most other synthetic data generation tools, metasyn strictly limits the statistical information in its data generation model to adhere to the highest privacy standards and only generates data that is similar on an individual column level. This makes it a great tool for initial exploration, code development, and sharing of datasets without compromising privacy at all - but it is not suitable for in-depth statistical analysis.

With metasyn, you fit a model to a dataset and synthesize data similar to the original based on that model. You can then export the synthetic data and the model used to generate it, in easy-to-read format. As a result, metasyn allows data owners to safely share synthetic datasets based on their source data, as well as the model used to generate it, without worrying about leaking any private information from the original dataset. 


Let's say you want to use metasyn to collaborate on a sensitive dataset with others. In this tutorial, we will show you everything you need to know to get started.

### Step 1: Setup

The first step is installing metasyn. The easiest way to do so is by installing it through `pip`. This can be done by typing the following command in your terminal:


```sh
pip install metasyn
```

Then, in a Python environment, you can import metasyn (and Polars, which will be used to load the dataset):


```python
import polars as pl
from metasyn import MetaFrame, VarSpec
```

### Step 2: Creating a DataFrame
Before we can pass a dataset into metasyn, we need to convert it to a [Polars](https://pola.rs) DataFrame. In doing so, we can indicate which columns contain categorical values. We can also tell polars to find columns that may contain dates or timestamps. Metasyn can later use this information to generate categorical or date-like values where appropriate. For more information on how to use Polars, check out the [Polars documentation](https://docs.pola.rs/).

For this tutorial, we will use the [Titanic dataset](https://www.kaggle.com/c/titanic/data), which comes preloaded with metasyn (its file path can be accessed using the `demo_file` function). We will specify the data types of the `Sex` and `Embarked` columns as categorical, and we will also try to parse dates in the DataFrame. 


```python
# Get the CSV file path for the Titanic dataset
from metasyn import demo_file
csv_path = demo_file("titanic") # Replace this with your file path if needed

# Create a Polars DataFrame
df = pl.read_csv(csv_path,
                 dtypes={"Sex": pl.Categorical,
                         "Embarked": pl.Categorical},
                 try_parse_dates=True)

```

### Step 3: Generating a MetaFrame
Now that we have created a DataFrame, we can easily generate a MetaFrame for it. Metasyn can later use this MetaFrame to generate synthetic data that aligns with the original dataset.

> A MetaFrame is essentially a model that captures the essentials of the original dataset (e.g., variable names, types, data types, the percentage of missing values, and distribution parameters), without containing any actual data entries. 


A MetaFrame can be created by simply calling `MetaFrame.fit_dataframe()`, passing in the DataFrame as a parameter.


```python
# Generate and fit a MetaFrame to the DataFrame 
mf = MetaFrame.fit_dataframe(df)
```

### Step 4: Generating synthetic data

With our MetaFrame in place, we can use it to generate synthetic data. To do so, we can call `synthesize` on our MetaFrame, and pass in the amount of rows of data that we want to generate. This will return a DataFrame with synthetic data, that is similar to our original dataset.


```python
# generate synthetic data
syn_df = mf.synthesize(5)
```

That's it! You can now read, analyze, modify, use and share this DataFrame as you would with any other - knowing that it does not contain any sensitive data!


### Step 5: Exporting the MetaFrame
Let's say we want to go one step further, and also share the MetaFrame alongside our synthetic data. We can easily do so by exporting it to a JSON file. 

These exported files follow the [Generative Metadata Format (GMF)](https://github.com/sodascience/generative_metadata_format). This is a format that was designed to be easy-to-read and understand. 

Other users can then import this file to generate synthetic data similar to the original dataset, without ever having access to the original data. In addition, due to these files being easy to read, others can easily understand and evaluate how the synthetic data is generated.

To export the MetaFrame, we can call the `export` method on an existing MetaFrame (in this case, `mf`), passing in the file path of where we want to save the JSON file.

```python
# Serialize and export the MetaFrame
mf.export("exported_metaframe.json")
```

To load the MetaFrame from the exported JSON file, we can use the `MetaFrame.from_json()` class method, passing in the file path as a parameter:


```python
# Create a MetaFrame based on a GMF (.json) file
mf = MetaFrame.from_json(file_path)
```

## Conclusion
You now know how to use metasyn to generate synthetic data from a dataset. Both the synthetic data and the model (MetaFrame) used to generate it can be shared safely, without compromising the privacy of the original data.

Enjoy using metasyn!

For more information on how to use metasyn, check out the [documentation](https://metasyn.readthedocs.io/en/latest/) or the [GitHub repository](https://github.com/sodascience/metasyn).
