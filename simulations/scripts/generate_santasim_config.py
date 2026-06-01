"""
Generate a SANTA-SIM XML config file for one pair of (mutation_rate, recombination_rate).
"""

import click


TEMPLATE = """<santa xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:noNamespaceSchemaLocation="santa.xsd">
\t<replicates>{num_replicates}</replicates>

\t<simulation>
\t\t<genome>
\t\t\t<length>29903</length>
\t\t\t<sequences file = "{reference_fasta}"></sequences>
\t\t</genome>

\t\t<!-- Neutral fitness -->
\t\t<fitnessFunction>
\t\t</fitnessFunction>

\t\t<population>
\t\t\t<populationSize>{population_size}</populationSize>
\t\t\t<inoculum>random</inoculum>
\t\t</population>

\t\t<mutator>
\t\t\t<nucleotideMutator>
\t\t\t\t<mutationRate>{mutation_rate}</mutationRate>
\t\t\t\t<transitionBias>1.0</transitionBias>
\t\t\t</nucleotideMutator>
\t\t</mutator>

\t\t<replicator>
\t\t\t<recombinantReplicator>
\t\t\t\t<dualInfectionProbability>0.0</dualInfectionProbability>
\t\t\t\t<recombinationProbability>{recombination_rate}</recombinationProbability>
\t\t\t</recombinantReplicator>
\t\t</replicator>

\t\t<epoch>
\t\t\t<generationCount>{generation_count}</generationCount>
\t\t</epoch>

\t\t<samplingSchedule>
\t\t\t<sampler>
\t\t\t\t<atFrequency>{at_frequency}</atFrequency>
\t\t\t\t<fileName>results_santasim/{sim_id}_rep%r.fasta</fileName>
\t\t\t\t<alignment>
\t\t\t\t\t<sampleSize>{sample_size}</sampleSize>
\t\t\t\t\t<format>FASTA</format>
\t\t\t\t\t<label>sample_r%r_g%g_i%s</label>
\t\t\t\t\t<breakpoints>TRUE</breakpoints>
\t\t\t\t</alignment>
\t\t\t</sampler>
\t\t</samplingSchedule>

\t</simulation>

</santa>
"""


def sim_id(mut_rate: str, rec_rate: str, at_frq: str) -> str:
    return f"seq_m{mut_rate}_r{rec_rate}_f{at_frq}"


@click.command()
@click.argument("output_xml", type=click.Path(dir_okay=False))
@click.option("--mutation-rate", required=True)
@click.option("--recombination-rate", required=True)
@click.option("--reference-fasta", required=True)
@click.option("--num-replicates", type=int, required=True)
@click.option("--generation-count", type=int, required=True)
@click.option("--at-frequency", required=True)
@click.option("--population-size", type=int, required=True)
@click.option("--sample-size", type=int, required=True)
def main(
    output_xml,
    mutation_rate,
    recombination_rate,
    reference_fasta,
    num_replicates,
    generation_count,
    at_frequency,
    population_size,
    sample_size,
):
    text = TEMPLATE.format(
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        reference_fasta=reference_fasta,
        num_replicates=num_replicates,
        generation_count=generation_count,
        at_frequency=at_frequency,
        population_size=population_size,
        sample_size=sample_size,
        sim_id=sim_id(
            mutation_rate,
            recombination_rate,
            at_frequency,
        ),
    )
    with open(output_xml, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
