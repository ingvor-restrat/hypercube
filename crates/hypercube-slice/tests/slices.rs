use hypercube_slice::{
    default_instrument_id, dot, top_abs, F64SliceReader, F64SliceWriter, LayoutRegistry,
    QuoteSliceReader, QuoteSliceWriter, QuoteV1, SliceCatalog, SliceCatalogEntry, TaqV1,
    TradeSliceReader, TradeSliceWriter, TradeV1, ValueType, TAQ_FLAG_HAS_QUOTE,
    TAQ_FLAG_QUOTE_STALE,
};

#[test]
fn layout_hash_and_symbol_lookup_work() {
    let symbols = vec!["AAPL".to_string(), "MSFT".to_string(), "NVDA".to_string()];
    let layout = LayoutRegistry::from_symbols("eq-us-test-v1", "eq", 8, &symbols).expect("layout");
    layout.validate().expect("valid layout");
    assert_eq!(layout.active_len(), 3);
    assert_eq!(layout.slot_for_symbol("msft").unwrap(), Some(1));
    assert_eq!(layout.symbol_for_slot(2), Some("NVDA"));
    assert_eq!(layout.entity_for_slot(2), Some("NVDA"));
    assert_eq!(layout.slot_for_entity("msft").unwrap(), Some(1));
    assert_eq!(layout.layout_hash().unwrap(), layout.layout_hash().unwrap());
}

#[test]
fn asset_class_defaults_are_not_equity_only() {
    let crypto = LayoutRegistry::from_symbols(
        "crypto-live-v1",
        "crypto",
        4,
        &["BTC-USD".to_string(), "ETH/USD".to_string()],
    )
    .expect("crypto layout");
    assert_eq!(crypto.asset_class, "crypto");
    assert_eq!(crypto.symbols[0].instrument_id, "crypto:BTC-USD");
    assert_eq!(crypto.symbols[1].instrument_id, "crypto:ETH/USD");

    let option = LayoutRegistry::from_symbols(
        "option-live-v1",
        "options",
        4,
        &["AAPL260117C00250000".to_string()],
    )
    .expect("option layout");
    assert_eq!(option.asset_class, "option");
    assert_eq!(
        option.symbols[0].instrument_id,
        "option:us:AAPL260117C00250000"
    );

    assert_eq!(
        default_instrument_id("future", "ESM6", Some("future:cme:")).unwrap(),
        "future:cme:ESM6"
    );
}

#[test]
fn f64_slice_create_write_read_and_dot() {
    let tmp = tempfile::tempdir().unwrap();
    let symbols = vec!["AAPL".to_string(), "MSFT".to_string(), "NVDA".to_string()];
    let layout = LayoutRegistry::from_symbols("eq-us-test-v1", "eq", 8, &symbols).expect("layout");
    let left_path = tmp.path().join("left.slice");
    let right_path = tmp.path().join("right.slice");

    let mut left = F64SliceWriter::create(&left_path, &layout, false).unwrap();
    let mut right = F64SliceWriter::create(&right_path, &layout, false).unwrap();
    left.write_slot(0, 1.0).unwrap();
    left.write_slot(1, 2.0).unwrap();
    left.write_slot(2, 3.0).unwrap();
    right.write_slot(0, 10.0).unwrap();
    right.write_slot(1, 20.0).unwrap();
    right.write_slot(2, 30.0).unwrap();
    left.flush().unwrap();
    right.flush().unwrap();

    let left_reader = F64SliceReader::open(&left_path).unwrap();
    let right_reader = F64SliceReader::open(&right_path).unwrap();
    left_reader.ensure_layout(&layout).unwrap();
    right_reader.ensure_layout(&layout).unwrap();
    assert_eq!(left_reader.value_at(1).unwrap(), 2.0);
    assert_eq!(left_reader.dot(&right_reader).unwrap(), 140.0);
    assert_eq!(dot(&[1.0, -2.0], &[3.0, 4.0]).unwrap(), -5.0);
}

#[test]
fn top_abs_is_bounded_and_deterministic() {
    let values = [f64::NAN, -4.0, 0.0, 4.0, 3.0, f64::INFINITY, -2.0];
    assert_eq!(top_abs(&values, 0), vec![]);
    assert_eq!(top_abs(&values, 2), vec![(1, -4.0), (3, 4.0)]);
    assert_eq!(
        top_abs(&values, usize::MAX),
        vec![(1, -4.0), (3, 4.0), (4, 3.0), (6, -2.0)]
    );
}

#[test]
fn writer_heartbeat_does_not_change_updated_timestamp() {
    let tmp = tempfile::tempdir().unwrap();
    let symbols = vec!["AAPL".to_string()];
    let layout = LayoutRegistry::from_symbols("eq-us-test-v1", "eq", 2, &symbols).expect("layout");
    let path = tmp.path().join("heartbeat.slice");

    let mut writer = F64SliceWriter::create(&path, &layout, false).unwrap();
    writer.write_slot(0, 1.0).unwrap();
    let before = writer.header();
    std::thread::sleep(std::time::Duration::from_millis(1));
    writer.heartbeat();
    writer.flush().unwrap();
    let after = F64SliceReader::open(&path).unwrap().header();

    assert!(after.heartbeat_ns > before.heartbeat_ns);
    assert_eq!(after.updated_ns, before.updated_ns);
}

#[test]
fn catalog_upsert_and_filter_work() {
    let mut catalog = SliceCatalog::default();
    catalog
        .upsert(SliceCatalogEntry {
            name: "crypto.latest.mid_px".to_string(),
            asset_class: "crypto".to_string(),
            layout_id: "crypto-live-v1".to_string(),
            layout_hash: 42,
            value_type: ValueType::F64,
            path: "/tmp/crypto.latest.mid_px.slice".to_string(),
            role: "latest".to_string(),
            description: None,
        })
        .unwrap();
    catalog
        .upsert(SliceCatalogEntry {
            name: "eq.latest.mid_px".to_string(),
            asset_class: "eq".to_string(),
            layout_id: "eq-live-v1".to_string(),
            layout_hash: 84,
            value_type: ValueType::F64,
            path: "/tmp/eq.latest.mid_px.slice".to_string(),
            role: "latest".to_string(),
            description: Some("equity midpoint".to_string()),
        })
        .unwrap();
    catalog.validate().unwrap();
    assert!(catalog.find("crypto.latest.mid_px").is_some());
    let crypto = catalog.by_asset_class("crypto").unwrap();
    assert_eq!(crypto.len(), 1);
    assert_eq!(crypto[0].name, "crypto.latest.mid_px");
}

#[test]
fn quote_trade_and_taq_struct_slices_work() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = LayoutRegistry::from_symbols(
        "eq-live-v1",
        "eq",
        4,
        &["AAPL".to_string(), "MSFT".to_string()],
    )
    .unwrap();
    let quote_path = tmp.path().join("quote.slice");
    let trade_path = tmp.path().join("trade.slice");

    let quote = QuoteV1::new(1_000, 1_100, 100.0, 200.0, 100.2, 300.0);
    let trade = TradeV1::new(1_500, 1_550, 1_600, 100.1, 25.0, Some("buy"));

    let mut quote_writer = QuoteSliceWriter::create(&quote_path, &layout, false).unwrap();
    let mut trade_writer = TradeSliceWriter::create(&trade_path, &layout, false).unwrap();
    quote_writer.write_slot(0, quote).unwrap();
    trade_writer.write_slot(0, trade).unwrap();
    quote_writer.flush().unwrap();
    trade_writer.flush().unwrap();

    let quote_reader = QuoteSliceReader::open(&quote_path).unwrap();
    let trade_reader = TradeSliceReader::open(&trade_path).unwrap();
    let read_quote = quote_reader.value_at(0).unwrap();
    let read_trade = trade_reader.value_at(0).unwrap();
    assert_eq!(read_quote.bid_px, 100.0);
    assert_eq!(read_trade.signed_qty, 25.0);

    let heartbeat_before = quote_reader.header().heartbeat_ns;
    std::thread::sleep(std::time::Duration::from_millis(1));
    quote_writer.heartbeat();
    quote_writer.flush().unwrap();
    assert!(quote_reader.header().heartbeat_ns > heartbeat_before);

    let taq = TaqV1::from_trade_and_quote(read_trade, Some(read_quote), 1_000);
    assert!(taq.flags & TAQ_FLAG_HAS_QUOTE != 0);
    assert_eq!(taq.quote_age_ns, 500);
    assert!(!taq.quote_is_stale());

    let stale = TaqV1::from_trade_and_quote(read_trade, Some(read_quote), 100);
    assert!(stale.flags & TAQ_FLAG_QUOTE_STALE != 0);
}

#[test]
fn layout_mismatch_blocks_dot() {
    let tmp = tempfile::tempdir().unwrap();
    let layout_a = LayoutRegistry::from_symbols(
        "layout-a",
        "eq",
        4,
        &["AAPL".to_string(), "MSFT".to_string()],
    )
    .unwrap();
    let layout_b = LayoutRegistry::from_symbols(
        "layout-b",
        "eq",
        4,
        &["MSFT".to_string(), "AAPL".to_string()],
    )
    .unwrap();
    let a_path = tmp.path().join("a.slice");
    let b_path = tmp.path().join("b.slice");
    F64SliceWriter::create(&a_path, &layout_a, false)
        .unwrap()
        .flush()
        .unwrap();
    F64SliceWriter::create(&b_path, &layout_b, false)
        .unwrap()
        .flush()
        .unwrap();
    let a = F64SliceReader::open(&a_path).unwrap();
    let b = F64SliceReader::open(&b_path).unwrap();
    assert!(a.dot(&b).is_err());
}
