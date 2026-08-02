use std::io::Result;

fn main() -> Result<()> {
    prost_build::Config::new()
        .out_dir("src/protos/")
        .compile_protos(
            &["schema/fileformat.proto", "schema/osmformat.proto"],
            &["schema/"],
        )?;

    Ok(())
}
